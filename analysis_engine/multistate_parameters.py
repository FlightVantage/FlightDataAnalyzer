# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import math
import numpy as np
import six

from pprint import pformat

from flightdatautilities import aircrafttables as at, dateext, units as ut

from hdfaccess.parameter import MappedArray

from analysis_engine.node import (
    A, M, P, S, helicopter, MultistateDerivedParameterNode
)

from analysis_engine.library import (
    align,
    all_of,
    any_of,
    calculate_flap,
    calculate_slat,
    clump_multistate,
    datetime_of_index,
    find_edges_on_state_change,
    including_transition,
    index_at_value,
    index_closest_value,
    first_valid_parameter,
    merge_masks,
    merge_two_parameters,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_masked_zeros_like,
    np_ma_zeros_like,
    offset_select,
    repair_mask,
    runs_of_ones,
    second_window,
    slice_duration,
    slices_and,
    slices_and_not,
    slices_from_to,
    slices_overlap,
    slices_remove_small_gaps,
    slices_remove_small_slices,
    smooth_signal,
    step_values,
    vstack_params,
    vstack_params_where_state,
)

from analysis_engine.settings import (
    MIN_CORE_RUNNING,
    MIN_FAN_RUNNING,
    MIN_FUEL_FLOW_RUNNING,
    REVERSE_THRUST_EFFECTIVE_EPR,
    REVERSE_THRUST_EFFECTIVE_N1
)

from flightdatautilities.numpy_utils import slices_int

logger = logging.getLogger(name=__name__)

class APEngaged(MultistateDerivedParameterNode):
    '''
    Determines if *any* of the "AP (*) Engaged" parameters are recording the
    state of Engaged.

    This is a discrete with only the Engaged state.
    '''

    name = 'AP Engaged'
    units = None
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):

        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.any(axis=0)
        self.array.mask = stacked.mask.any(axis=0)


class APChannelsEngaged(MultistateDerivedParameterNode):
    '''
    Assess the number of autopilot systems engaged.

    Airbus and Boeing = 1 autopilot at a time except when "Land" mode
    selected when 2 (Dual) or 3 (Triple) can be engaged. Airbus favours only
    2 APs, Boeing is happier with 3 though some older types may only have 2.
    '''
    name = 'AP Channels Engaged'
    units = None
    values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}

    @classmethod
    def can_operate(cls, available):
        return len(available) >= 2

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):
        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.sum(axis=0)
        self.offset = offset_select('mean', [ap1, ap2, ap3])


class Configuration(MultistateDerivedParameterNode):
    '''
    Parameter for aircraft that use configuration. Reflects the actual state
    of the aircraft. See "Flap Lever" or "Flap Lever (Synthetic)" which show
    the physical lever detents selectable by the crew.

    Multi-state with the following mapping::
        %s

    Some values are based on footnotes in various pieces of documentation:
    - 2(a) corresponds to CONF 1*
    - 3(b) corresponds to CONF 2*

    Note: Does not use the Flap Lever position. This parameter reflects the
    actual configuration state of the aircraft rather than the intended state
    represented by the selected lever position.
    ''' % pformat(at.constants.AVAILABLE_CONF_STATES)
    values_mapping = at.constants.AVAILABLE_CONF_STATES
    align_frequency = 2
    units = None

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer'),
                    model=A('Model'), series=A('Series'), family=A('Family'),):

        if manufacturer and not manufacturer.value == 'Airbus':
            return False

        if family and family.value in ('A300', 'A310',):
            return False

        if not all_of(('Slat Including Transition', 'Flap Including Transition', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_conf_angles(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No conf angles available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=M('Flap Including Transition'), slat=M('Slat Including Transition'),
               model=A('Model'), series=A('Series'), family=A('Family'),):

        angles = at.get_conf_angles(model.value, series.value, family.value)

        # initialize an empty masked array the same length as flap array
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array, dtype=np.short),
                                 values_mapping=self.values_mapping)

        for (state, (s, f, a)) in six.iteritems(angles):
            condition = (flap.array == f)
            if s is not None:
                condition &= (slat.array == s)

            self.array[condition] = state

        nearest_neighbour_mask_repair(self.array, copy=False,
                                      repair_gap_size=(30 * self.hz),
                                      direction='backward')


class Daylight(MultistateDerivedParameterNode):
    '''
    Calculate Day or Night based upon Civil Twilight.

    FAA Regulation FAR 1.1 defines night as: "Night means the time between
    the end of evening civil twilight and the beginning of morning civil
    twilight, as published in the American Air Almanac, converted to local
    time.

    EASA EU OPS 1 Annex 1 item (76) states: 'night' means the period between
    the end of evening civil twilight and the beginning of morning civil
    twilight or such other period between sunset and sunrise as may be
    prescribed by the appropriate authority, as defined by the Member State;

    CAA regulations confusingly define night as 30 minutes either side of
    sunset and sunrise, then include a civil twilight table in the AIP.

    With these references, it was decided to make civil twilight the default.
    '''
    align = True
    # 1/4 is the minimum allowable frequency due to minimum data boundary
    # of 4 seconds.
    align_frequency = 1 / 4.0
    align_offset = 0.0
    units = None

    values_mapping = {
        0: 'Night',
        1: 'Day'
    }

    def derive(self,
               latitude=P('Latitude Smoothed'),
               longitude=P('Longitude Smoothed'),
               start_datetime=A('Start Datetime'),
               duration=A('HDF Duration')):
        # Set default to 'Day'
        array_len = int(duration.value * self.frequency)
        self.array = np.ma.ones(array_len)
        for step in range(array_len):
            curr_dt = datetime_of_index(start_datetime.value, step, 1)
            lat = latitude.array[step]
            lon = longitude.array[step]
            if lat and lon:
                if not dateext.is_day(curr_dt, lat, lon):
                    # Replace values with Night
                    self.array[step] = 0
                else:
                    continue  # leave array as 1
            else:
                # either is masked or recording 0.0 which is invalid too
                self.array[step] = np.ma.masked


class EngRunning(object):
    '''
    Abstract class for inheriting by EngRunning derived parameters.
    '''
    engnum = 0  # Replace with '2' for Eng (2)
    units = None
    values_mapping = {
        0: 'Not Running',
        1: 'Running',
    }
    # Workaround for NotImplementedError: Unknown Type raised from
    # process_flight
    node_type = MultistateDerivedParameterNode

    @classmethod
    def can_operate(cls, available):
        return 'Eng (%d) N1' % cls.engnum in available or \
               'Eng (%d) N2' % cls.engnum in available or \
               'Eng (%d) Np' % cls.engnum in available or \
               'Eng (%d) Fuel Flow' % cls.engnum in available

    def determine_running(self, eng_n1, eng_n2, eng_np, fuel_flow, ac_type):
        '''
        TODO: Include Fuel cut-off switch if recorded?
        TODO: Confirm that all engines were recording for the N2 Min / Fuel Flow
        Min parameters - theoretically there could be only three engines in the
        frame for a four engine aircraft. Use "Engine Count".
        '''
        if eng_np:
            # If it's got propellors, this overrides core engine measurements.
            return np.ma.where(eng_np.array > MIN_FAN_RUNNING, 'Running', 'Not Running')
        elif eng_n2 or fuel_flow and ac_type != helicopter:
            # Ideally have N2 and Fuel Flow with both available,
            # otherwise use just one source
            n2_running = eng_n2.array > MIN_CORE_RUNNING if eng_n2 \
                else np.zeros_like(fuel_flow.array, dtype=bool)
            fuel_flowing = fuel_flow.array > MIN_FUEL_FLOW_RUNNING if fuel_flow \
                else np.zeros_like(eng_n2.array, dtype=bool)
            data = n2_running.data | fuel_flowing.data
            mask = n2_running.mask & fuel_flowing.mask
            return np.ma.where(np.ma.array(data, mask=mask), 'Running', 'Not Running')
        else:
            # Fall back on N1
            return np.ma.where(eng_n1.array > MIN_FAN_RUNNING, 'Running', 'Not Running')


class Eng1Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 1
    name = 'Eng (1) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (1) N1'),
               eng_n2=P('Eng (1) N2'),
               eng_np=P('Eng (1) Np'),
               fuel_flow=P('Eng (1) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng2Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 2
    name = 'Eng (2) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (2) N1'),
               eng_n2=P('Eng (2) N2'),
               eng_np=P('Eng (2) Np'),
               fuel_flow=P('Eng (2) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng3Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 3
    name = 'Eng (3) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (3) N1'),
               eng_n2=P('Eng (3) N2'),
               eng_np=P('Eng (3) Np'),
               fuel_flow=P('Eng (3) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng4Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 4
    name = 'Eng (4) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (4) N1'),
               eng_n2=P('Eng (4) N2'),
               eng_np=P('Eng (4) Np'),
               fuel_flow=P('Eng (4) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng_AllRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when all available engines are running.
    '''
    name = 'Eng (*) All Running'
    units = None

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Eng (*) N1 Min' in available
        else:
            return 'Eng (*) N1 Min' in available or \
                   'Eng (*) N2 Min' in available or \
                   'Eng (*) Np Min' in available or \
                   'Eng (*) Fuel Flow Min' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Min'),
               eng_n2=P('Eng (*) N2 Min'),
               eng_np=P('Eng (*) Np Min'),
               fuel_flow=P('Eng (*) Fuel Flow Min'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng_AnyRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when any engines are running.

    This is useful with 'Eng (*) All Running' to detect if not all engines are
    running.
    '''
    name = 'Eng (*) Any Running'
    units = None

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Eng (*) N1 Max' in available
        else:
            return 'Eng (*) N1 Max' in available or \
                   'Eng (*) N2 Max' in available or \
                   'Eng (*) Np Max' in available or \
                   'Eng (*) Fuel Flow Max' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Max'),
               eng_n2=P('Eng (*) N2 Max'),
               eng_np=P('Eng (*) Np Max'),
               fuel_flow=P('Eng (*) Fuel Flow Max'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Flap(MultistateDerivedParameterNode):
    '''
    Steps raw Flap angle from surface into detents.
    '''

    units = ut.DEGREE
    # Currently uses the frequency of the Flap Angle parameter - might
    # consider upsampling to 2Hz for the Kernal sizes in the calculate_flap
    # function
    ##align_frequency = 2

    @classmethod
    def can_operate(cls, available, frame=A('Frame'),
                    model=A('Model'), series=A('Series'), family=A('Family')):

        frame_name = frame.value if frame else None
        family_name = family.value if family else None

        if frame_name == 'L382-Hercules' or family_name == 'C208':
            return 'Altitude AAL' in available

        if family_name == 'Citation VLJ':
            return all_of(('HDF Duration', 'Landing', 'Takeoff'), available)

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            #Q: Everyone should have a flap map - so raise error?
            cls.exception("No flap mapping available for '%s', '%s', '%s'.",
                          model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family'),
               frame=A('Frame'),
               hdf_duration=A('HDF Duration')):

        family_name = family.value if family else None

        if 'B737' in family_name:
            _slices = runs_of_ones(np.logical_and(flap.array>=0.9, flap.array<=2.1))
            for s in _slices:
                flap.array[s] = smooth_signal(flap.array[s], window_len=5, window='flat')
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'lever', flap, model, series, family)


class FlapIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return all_of(('Flap', 'Model', 'Series', 'Family'), available)

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_angle=P('Flap Angle'), flap=M('Flap'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping = at.get_flap_map(model.value, series.value, family.value)
        if flap_angle:
            self.array = including_transition(flap_angle.array, self.values_mapping, hz=self.hz)
        else:
            # if we do not have flap angle use flap, use states as values
            # will vary between frames
            array = MappedArray(np_ma_masked_zeros_like(flap.array),
                                values_mapping=self.values_mapping)
            for value, state in six.iteritems(self.values_mapping):
                array[flap.array == state] = state
            self.array = array


class FlapExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_angle=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        family_name = family.value if family else None
        if "B737" in family_name:
            _slices = runs_of_ones(np.logical_and(flap_angle.array>=0.9, flap_angle.array<=2.1))
            for s in _slices:
                flap_angle.array[s] = smooth_signal(flap_angle.array[s], window_len=5, window='flat')
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'excluding', flap_angle, model, series, family)


class Flaperon(MultistateDerivedParameterNode):
    '''
    Where Ailerons move together and used as Flaps, these are known as
    "Flaperon" control.

    Flaperons are measured where both Left and Right Ailerons move down,
    which on the left creates possitive roll but on the right causes negative
    roll. The difference of the two signals is the Flaperon control.

    The Flaperon is stepped at the start of movement into the nearest aileron
    detents, e.g. 0, 5, 10 deg

    Note: This is used for Airbus models and does not necessarily mean as
    much to other aircraft types.
    '''
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Aileron (L)', 'Aileron (R)', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_aileron_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No aileron/flaperon mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, al=P('Aileron (L)'), ar=P('Aileron (R)'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        # Take the difference of the two signals (which should cancel each
        # other out when rolling) and divide the range by two (to account for
        # the left going negative and right going positive when flaperons set)
        #al.array = (al.array - ar.array) / 2

        #self.values_mapping = at.get_aileron_map(model.value, series.value, family.value)
        #self.array, self.frequency, self.offset = calculate_surface_angle(
            #'lever',
            #al,
            #self.values_mapping.keys(),
        #)
        flaperon_angle = (al.array - ar.array) / 2
        self.values_mapping = at.get_aileron_map(model.value, series.value, family.value)
        self.array = step_values(flaperon_angle,
                                 self.values_mapping.keys(),
                                 al.hz, step_at='move_start')


class Slat(MultistateDerivedParameterNode):
    '''
    Steps raw slat angle into detents.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'lever',
            slat,
            model,
            series,
            family,
        )


class SlatExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the lower of the start and endpoints of the movement
    apply. This minimises the chance of needing a slat overspeed inspection.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'excluding',
            slat,
            model,
            series,
            family,
        )


class SlatIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a slat overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping = at.get_slat_map(model.value, series.value, family.value)
        self.array=including_transition(slat.array, self.values_mapping, hz=self.hz)


class SpeedbrakeDeployed(MultistateDerivedParameterNode):
    '''
    Follows same logic as when deriving speedbaker from spoiler angles two
    matching spoilers in deployed state indicates speedbrake, single side
    depolyment indicates roll.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Deployed'}

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        if family and family.value == 'B787':
            return 'Speedbrake Handle' in available

        return 'Spoiler Deployed' in available or \
               all_of(('Spoiler (L) Deployed', 'Spoiler (R) Deployed'), available) or \
               all_of(('Spoiler (L) (1) Deployed', 'Spoiler (R) (1) Deployed'), available) or \
               all_of(('Spoiler (L) (2) Deployed', 'Spoiler (R) (2) Deployed'), available) or \
               all_of(('Spoiler (L) (3) Deployed', 'Spoiler (R) (3) Deployed'), available) or \
               all_of(('Spoiler (L) (4) Deployed', 'Spoiler (R) (4) Deployed'), available) or \
               all_of(('Spoiler (L) (5) Deployed', 'Spoiler (R) (5) Deployed'), available) or \
               all_of(('Spoiler (L) (6) Deployed', 'Spoiler (R) (6) Deployed'), available) or \
               all_of(('Spoiler (L) (7) Deployed', 'Spoiler (R) (7) Deployed'), available) or \
               all_of(('Spoiler (L) Outboard Deployed', 'Spoiler (R) Outboard Deployed'), available) or \
               'Spoiler' in available or \
               all_of(('Spoiler (L)', 'Spoiler (R)'), available) or \
               all_of(('Spoiler (L) (1)', 'Spoiler (R) (1)'), available) or \
               all_of(('Spoiler (L) (2)', 'Spoiler (R) (2)'), available) or \
               all_of(('Spoiler (L) (3)', 'Spoiler (R) (3)'), available) or \
               all_of(('Spoiler (L) (4)', 'Spoiler (R) (4)'), available) or \
               all_of(('Spoiler (L) (5)', 'Spoiler (R) (5)'), available) or \
               all_of(('Spoiler (L) (6)', 'Spoiler (R) (6)'), available) or \
               all_of(('Spoiler (L) (7)', 'Spoiler (R) (7)'), available) or \
               all_of(('Spoiler (L) Outboard', 'Spoiler (R) Outboard'), available)

    def derive(self, dep=M('Spoiler Deployed'),
               ld=M('Spoiler (L) Deployed'),
               rd=M('Spoiler (R) Deployed'),
               l1d=M('Spoiler (L) (1) Deployed'),
               l2d=M('Spoiler (L) (2) Deployed'),
               l3d=M('Spoiler (L) (3) Deployed'),
               l4d=M('Spoiler (L) (4) Deployed'),
               l5d=M('Spoiler (L) (5) Deployed'),
               l6d=M('Spoiler (L) (6) Deployed'),
               l7d=M('Spoiler (L) (7) Deployed'),
               r1d=M('Spoiler (R) (1) Deployed'),
               r2d=M('Spoiler (R) (2) Deployed'),
               r3d=M('Spoiler (R) (3) Deployed'),
               r4d=M('Spoiler (R) (4) Deployed'),
               r5d=M('Spoiler (R) (5) Deployed'),
               r6d=M('Spoiler (R) (6) Deployed'),
               r7d=M('Spoiler (R) (7) Deployed'),
               loutd=M('Spoiler (L) Outboard Deployed'),
               routd=M('Spoiler (R) Outboard Deployed'),
               spoiler=P('Spoiler'),
               l=P('Spoiler (L)'),
               r=P('Spoiler (R)'),
               l1=P('Spoiler (L) (1)'),
               l2=P('Spoiler (L) (2)'),
               l3=P('Spoiler (L) (3)'),
               l4=P('Spoiler (L) (4)'),
               l5=P('Spoiler (L) (5)'),
               l6=P('Spoiler (L) (6)'),
               l7=P('Spoiler (L) (7)'),
               r1=P('Spoiler (R) (1)'),
               r2=P('Spoiler (R) (2)'),
               r3=P('Spoiler (R) (3)'),
               r4=P('Spoiler (R) (4)'),
               r5=P('Spoiler (R) (5)'),
               r6=P('Spoiler (R) (6)'),
               r7=P('Spoiler (R) (7)'),
               lout=P('Spoiler (L) Outboard'),
               rout=P('Spoiler (R) Outboard'),
               handle=P('Speedbrake Handle'),
               family=A('Family')):

        left = (ld, l1d, l2d, l3d, l4d, l5d, l6d, l7d, loutd,
                l, l1, l2, l3, l4, l5, l6, l7, lout)
        right = (rd, r1d, r2d, r3d, r4d, r5d, r6d, r7d, routd,
                 r, r1, r2, r3, r4, r5, r6, r7, rout)
        pairs = list(zip(left, right))
        state = 'Deployed'

        def is_deployed(param):
            if not param:
                return
            array = np_ma_zeros_like(
                param.array, dtype=np.bool, mask=param.array.mask)
            if state in param.name:
                if state not in param.array.state:
                    logger.warning("State '%s' not found in param '%s'", state, param.name)
                    return
                matching = param.array == state
            elif family and family.value == 'MD-11':
                matching = param.array >= 15
            else:
                matching = param.array >= 10

            for s in runs_of_ones(matching, min_samples=1):
                array[s] = True
            return array

        if family and family.value == 'B787':
            speedbrake = np.ma.zeros(len(handle.array), dtype=np.short)
            stepped_array = step_values(handle.array, [0, 20], hz=handle.hz)
            speedbrake[stepped_array == 20] = 1
            self.array = speedbrake
        else:
            combined = [a for a in (is_deployed(p) for p in (dep, spoiler)) if a is not None]
            combined.extend(
                np.ma.vstack(arrays).all(axis=0) for arrays in
                ([is_deployed(p) for p in pair] for pair in pairs)
                if all(a is not None for a in arrays))

            if not combined:
                self.array = np_ma_zeros_like(
                    next(p.array for p in (dep, spoiler) + left + right if p is not None),
                    dtype=np.short, mask=True)
                return

            stack = np.ma.vstack(combined)

            array = np_ma_zeros_like(stack[0], dtype=np.short)
            array = np.ma.where(stack.any(axis=0), 1, array)

            # mask indexes with greater than 50% masked values
            mask = np.ma.where(stack.mask.sum(axis=0).astype(float) / len(stack) * 100 > 50, 1, 0)
            self.array = array
            self.array.mask = mask


class StableApproachStages(object):
    '''
    During the Descent and Approach, the following steps are assessed in turn
    to determine the aircraft stability:

    1. Gear is down
    2. Landing Flap is set
    3. Track is aligned to Runway (within 12 degrees or 30 if offset approach)
    4. Airspeed:
        - airspeed minus selected approach speed within -5 to +15 knots (for 3 secs)
        - or Vapp within -5 to +10 knots (for 3 secs)
        - or Vref within -5 to +35 knots (for 3 secs)
    5. Glideslope deviation within 1 dot
    6. Localizer deviation within 1 dot
    7. Vertical speed between -1100 and -200 fpm
    8. Engine Thrust greater than 40% N1 or 35% (A319/B787) or 1.09 EPR
       (for 10 secs) or 1.02 (A319, A320, A321)

    if all the above steps are met, the result is the declaration of:
    9. "Stable"

    Notes:

    Airspeed is relative to "Airspeed Selected" where available as this will
    account for the reference speed and any compensation for the wind speed.

    If Vapp is recorded, a more constraint airspeed threshold is applied.

    Where parameters are not monitored below a certain threshold (e.g. ILS
    below 200ft) the stability criteria just before 200ft is reached is
    continued through to landing. So if one was unstable due to ILS
    Glideslope down to 200ft, that stability is assumed to continue through
    to landing.

    TODO/REVIEW:
    ============
    * Check for 300ft limit if turning onto runway late and ignore stability
      criteria before this? Alternatively only assess criteria when heading is
      within 50.
    * Add hysteresis (3 second gliding windows for GS / LOC etc.)
    * Engine cycling check
    * Use Engine TPR for B787 instead of EPR if available.
    '''
    units = None
    values_mapping = {
        0: '-',  # All values should be masked anyway, this helps align values
        1: 'Gear Not Down',
        2: 'Not Landing Flap',
        3: 'Track Not Aligned',
        4: 'Aspd Not Stable',  # Q: Split into two Airspeed High/Low?
        5: 'GS Not Stable',
        6: 'Loc Not Stable',
        7: 'IVV Not Stable',
        8: 'Eng Thrust Not Stable',
        9: 'Stable',
    }

    align_frequency = 1

    def derive_stable_approach(self, apps, phases, gear, flap, tdev, aspd_rel,
                               aspd_minus_sel, vspd, gdev, ldev, eng_n1,
                               eng_epr, alt, vapp, family, model):

        # create an empty fully masked array
        self.array = np.ma.zeros(len(alt.array), dtype=np.short)
        self.array.mask = True

        self.repair = lambda ar, ap, method='interpolate': repair_mask(
            ar[ap], raise_entirely_masked=False, method=method)
        for approach in apps:
            # lookup descent from approach, dont zip as not guanenteed to have the same
            # number of descents and approaches
            phase = phases.get_last(within_slice=approach.slice, within_use='any')
            if not phase:
                continue
            # use Combined descent phase slice as it contains the data from
            # top of descent to touchdown (approach starts and finishes later)
            approach.slice = phase.slice

            # FIXME: approaches shorter than 10 samples will not work due to
            # the use of moving_average with a width of 10 samples.
            if approach.slice.stop - approach.slice.start < 10:
                continue
            # Restrict slice to 10 seconds after landing if we hit the ground
            gnd = index_at_value(alt.array, 0, approach.slice)
            if gnd and gnd + 10 < approach.slice.stop:
                stop = gnd + 10
            else:
                stop = approach.slice.stop
            _slice = slices_int(approach.slice.start, stop)

            altitude = self.repair(alt.array, _slice)
            index_at_50 = int(index_closest_value(altitude, 50))
            index_at_200 = int(index_closest_value(altitude, 200))

            if gear:
                #== 1. Gear Down ==
                stable = self._stable_1_gear_down(_slice, gear)
            if flap:
                #== 2. Landing Flap ==
                stable = self._stable_2_landing_flap(_slice, stable, altitude, flap)
            if tdev:
                #== 3. Track Deviation ==
                stable = self._stable_3_track_deviation(_slice, stable, approach, tdev)
            if aspd_rel or aspd_minus_sel:
                #== 4. Airspeed Relative ==
                stable = self._stable_4_airspeed_relative(_slice, stable, aspd_minus_sel, aspd_rel, vapp, index_at_50, altitude)
            if gdev:
                #== 5. Glideslope Deviation ==
                stable = self._stable_5_glideslope_deviation(_slice, stable, approach, gdev, index_at_200, altitude)
            if ldev:
                #== 6. Localizer Deviation ==
                stable = self._stable_6_localizer_deviation(_slice, stable, approach, ldev, index_at_200, altitude)
            if vspd:
                #== 7. Vertical Speed ==
                stable = self._stable_7_vertical_speed(_slice, stable, approach, vspd, index_at_50, altitude)
            if eng_epr or eng_n1:
                #== 8. Engine Thrust (N1/EPR) ==
                stable = self._stable_8_engine_thrust(_slice, stable, eng_epr, eng_n1, family, index_at_50, altitude)
            #== 9. Stable ==
            # Congratulations; whatever remains in this approach is stable!
            self.array[_slice][stable] = 9

        #endfor
        return

    def _stable_1_gear_down(self, _slice, gear):
        #== 1. Gear Down ==
        # prepare data for this appproach:
        gear_down = self.repair(gear.array, _slice, method='fill_start')
        # Assume unstable due to Gear Down at first
        self.array[_slice] = 1
        landing_gear_set = (gear_down == 'Down')
        stable = landing_gear_set.filled(True)  # assume stable (gear down)
        return stable

    def _stable_2_landing_flap(self, _slice, stable, altitude, flap):
        #== 2. Landing Flap ==
        # prepare data for this appproach:
        flap_lever = self.repair(flap.array, _slice, method='fill_start')
        # not due to landing gear so try to prove it wasn't due to Landing Flap
        self.array[_slice][stable] = 2
        # look for maximum flap used in approach below 1,000ft, otherwise
        # go-arounds can detect the start of flap retracting as the
        # landing flap.
        landing_flap = np.ma.where(altitude < 1000, flap_lever, np.ma.masked).max()
        if landing_flap is np.ma.masked:
            # try looking above 1000ft
            landing_flap = np.ma.where(altitude > 1000, flap_lever, np.ma.masked).max()

        if landing_flap is not np.ma.masked:
            landing_flap_set = (flap_lever == landing_flap)
            # assume stable (flap set)
            stable &= landing_flap_set.filled(True)
        else:
            # All landing flap is masked, assume stable
            logger.warning(
                'StableApproach: the landing flap is all masked in '
                'the approach.')
            stable &= True
        return stable

    def _stable_3_track_deviation(self, _slice, stable, approach, tdev):
        #== 3. Track Deviation ==
        # prepare data for this appproach:
        track_dev = self.repair(tdev.array, _slice)

        self.array[_slice][stable] = 3
        runway = approach.approach_runway
        if runway and runway.get('localizer', {}).get('is_offset'):
            # offset ILS Localizer or offset approach without ILS (IAN approach)
            STABLE_TRACK = 30  # degrees
        else:
            # use 12 to allow rolling a little over the 10 degrees when
            # aligning to runway.
            STABLE_TRACK = 12  # degrees
        stable_track_dev = abs(track_dev) <= STABLE_TRACK
        stable &= stable_track_dev.filled(True)  # assume stable (on track)
        return stable

    def _stable_4_airspeed_relative(self, _slice, stable, aspd_minus_sel, aspd_rel, vapp, index_at_50, altitude):
        #== 4. Airspeed Relative ==
        # prepare data for this appproach:
        if aspd_minus_sel:
            airspeed = self.repair(aspd_minus_sel.array, _slice)
        elif aspd_rel:
            airspeed = self.repair(aspd_rel.array, _slice)
        else:
            airspeed = None

        if airspeed is not None:
            self.array[_slice][stable] = 4
            if aspd_minus_sel:
                # Airspeed relative to selected speed
                if aspd_rel:
                    low_limit_airspeed = self.repair(aspd_rel.array, _slice)
                else:
                    low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 15
            elif vapp:
                # Those aircraft which record a variable Vapp shall have more constraint thresholds
                low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 10
            else:
                # Most aircraft record only Vref - as we don't know the wind correction be more lenient
                low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 35

            stable_airspeed = (low_limit_airspeed >= STABLE_AIRSPEED_BELOW_REF) & (airspeed <= STABLE_AIRSPEED_ABOVE_REF)
            # extend the stability at the end of the altitude threshold through to landing
            stable_airspeed[altitude < 50] = stable_airspeed[index_at_50]
            stable &= stable_airspeed.filled(True)  # if no V Ref speed, values are masked so consider stable as one is not flying to the vref speed??
        return stable

    def _stable_5_glideslope_deviation(self, _slice, stable, approach, gdev, index_at_200, altitude):
        #== 5. Glideslope Deviation ==
        # prepare data for this appproach:
        glideslope = self.repair(gdev.array, _slice) if gdev else None  # optional
        if approach.gs_est:
            self.array[_slice][stable] = 5
            STABLE_GLIDESLOPE = 1.0  # dots
            stable_gs = (abs(glideslope) <= STABLE_GLIDESLOPE)
            # extend the stability at the end of the altitude threshold through to landing
            stable_gs[altitude < 200] = stable_gs[index_at_200]
            stable &= stable_gs.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
        return stable

    def _stable_6_localizer_deviation(self, _slice, stable, approach, ldev, index_at_200, altitude):
        #== 6. Localizer Deviation ==
        # prepare data for this appproach:
        localizer = self.repair(ldev.array, _slice) if ldev else None  # optional

        if approach.gs_est and approach.loc_est:
            self.array[_slice][stable] = 6
            STABLE_LOCALIZER = 1.0  # dots
            stable_loc = (abs(localizer) <= STABLE_LOCALIZER)
            # extend the stability at the end of the altitude threshold through to landing
            stable_loc[altitude < 200] = stable_loc[index_at_200]
            stable &= stable_loc.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
        return stable

    def _stable_7_vertical_speed(self, _slice, stable, approach, vspd, index_at_50, altitude, ):
        #== 7. Vertical Speed ==
        # prepare data for this appproach:
        # apply quite a large moving average to smooth over peaks and troughs
        vertical_speed = moving_average(self.repair(vspd.array, _slice), 11)
        runway = approach.approach_runway

        self.array[_slice][stable] = 7
        STABLE_VERTICAL_SPEED_MAX = -200
        STABLE_VERTICAL_SPEED_MIN = -1100
        if runway:
            gs_angle = runway.get('glideslope', {}).get('angle')
            # offset ILS Localizer or offset approach without ILS (IAN approach)
            if gs_angle is not None and gs_angle > 3:
                STABLE_VERTICAL_SPEED_MIN = -1500
        stable_vert = (vertical_speed >= STABLE_VERTICAL_SPEED_MIN) & (vertical_speed <= STABLE_VERTICAL_SPEED_MAX)
        # extend the stability at the end of the altitude threshold through to landing
        stable_vert[altitude < 50] = stable_vert[index_at_50]
        stable &= stable_vert.filled(True)
        return stable

    def _stable_8_engine_thrust(self, _slice, stable, eng_epr, eng_n1, family, index_at_50, altitude):
        #== 8. Engine Thrust (N1/EPR) ==
        if eng_epr:
            # use EPR if available
            engine = self.repair(eng_epr.array, _slice)
        else:
            engine = self.repair(eng_n1.array, _slice)

        self.array[_slice][stable] = 8
        # Patch this value depending upon aircraft type
        if eng_epr:
            if family and family.value in ('A319', 'A320', 'A321'):
                STABLE_EPR_MIN = 1.02  # Ratio
            else:
                STABLE_EPR_MIN = 1.09  # Ratio
            stable_engine = (engine >= STABLE_EPR_MIN)
        else:
            if family and family.value in ('B787', 'A319'):
                STABLE_N1_MIN = 35  # %
            else:
                STABLE_N1_MIN = 40  # %
            stable_engine = (engine >= STABLE_N1_MIN)
        # extend the stability at the end of the altitude threshold through to landing
        stable_engine[altitude < 50] = stable_engine[index_at_50]
        stable &= stable_engine.filled(True)
        return stable


class StableApproach(MultistateDerivedParameterNode, StableApproachStages):
    units = None

    @classmethod
    def can_operate(cls, available):
        # Many parameters are optional dependencies
        deps = ['Approach Information', 'Descent', 'Gear Down', 'Flap',
                'Vertical Speed', 'Altitude AAL',]
        return all_of(deps, available) and (
            'Eng (*) N1 Avg For 10 Sec' in available or
            'Eng (*) EPR Avg For 10 Sec' in available)

    def derive(self,
               apps=A('Approach Information'),
               phases=S('Descent'),
               gear=M('Gear Down'),
               flap=M('Flap'),
               tdev=P('Track Deviation From Runway'),
               aspd_rel=P('Airspeed Relative For 3 Sec'),
               aspd_minus_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               vspd=P('Vertical Speed'),
               gdev=P('ILS Glideslope'),
               ldev=P('ILS Localizer'),
               eng_n1=P('Eng (*) N1 Avg For 10 Sec'),
               eng_epr=P('Eng (*) EPR Avg For 10 Sec'),
               alt=P('Altitude AAL'),
               vapp=P('Vapp'),
               family=A('Family'),
               model=A('Model')):
        self.derive_stable_approach(apps, phases, gear, flap, tdev, aspd_rel,
                                    aspd_minus_sel, vspd, gdev, ldev, eng_n1,
                                    eng_epr, alt, vapp, family, model)
