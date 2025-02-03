# -*- coding: utf-8 -*-

from __future__ import print_function

import geomag
import numpy as np
import six

from copy import deepcopy
from datetime import date
from math import radians
from operator import attrgetter
from numpy import interp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import medfilt

from flightdatautilities import aircrafttables as at, units as ut

from analysis_engine.exceptions import DataFrameError

from analysis_engine.node import (
    A, M, P, S, KPV, KTI, aeroplane, App, DerivedParameterNode, helicopter
)

from analysis_engine.library import (
    air_track,
    align,
    all_of,
    any_of,
    alt2press,
    alt2sat,
    alt_dev2alt,
    bearing_and_distance,
    bearings_and_distances,
    blend_parameters,
    blend_two_parameters,
    cas2dp,
    coreg,
    cycle_finder,
    dp2tas,
    dp_over_p2mach,
    filter_vor_ils_frequencies,
    first_valid_parameter,
    first_valid_sample,
    first_order_lag,
    first_order_washout,
    from_isa,
    ground_track,
    ground_track_precise,
    groundspeed_from_position,
    heading_diff,
    hysteresis,
    integrate,
    ils_localizer_align,
    index_at_value,
    interpolate,
    is_index_within_slice,
    last_valid_sample,
    latitudes_and_longitudes,
    localizer_scale,
    lookup_table,
    machsat2tat,
    machtat2sat,
    mask_inside_slices,
    mask_outside_slices,
    match_altitudes,
    max_value,
    mb2ft,
    merge_masks,
    most_common_value,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_ones_like,
    np_ma_masked_zeros,
    np_ma_masked_zeros_like,
    np_ma_zeros_like,
    offset_select,
    overflow_correction,
    peak_curvature,
    press2alt,
    power_floor,
    rate_of_change,
    rate_of_change_array,
    repair_mask,
    rms_noise,
    runs_of_ones,
    runway_deviation,
    runway_distances,
    runway_heading,
    runway_length,
    runway_snap_dict,
    second_window,
    shift_slice,
    slices_and,
    slices_of_runs,
    slices_between,
    slices_find_small_slices,
    slices_from_to,
    slices_from_ktis,
    slices_not,
    slices_or,
    slices_remove_small_slices,
    slices_split,
    smooth_track,
    straighten_altitudes,
    straighten_headings,
    track_linking,
    value_at_index,
    vstack_params
)

from analysis_engine.settings import (
    AIRSPEED_THRESHOLD,
    ALTITUDE_RADIO_OFFSET_LIMIT,
    ALTITUDE_RADIO_MAX_RANGE,
    ALTITUDE_AAL_TRANS_ALT,
    ALTITUDE_AGL_SMOOTHING,
    ALTITUDE_AGL_TRANS_ALT,
    AZ_WASHOUT_TC,
    BOUNCED_LANDING_THRESHOLD,
    CLIMB_THRESHOLD,
    HYSTERESIS_FPROC,
    GRAVITY_IMPERIAL,
    GRAVITY_METRIC,
    LANDING_THRESHOLD_HEIGHT,
    MIN_VALID_FUEL,
    VERTICAL_SPEED_LAG_TC
)

from flightdatautilities.numpy_utils import (
    slices_int,
    np_ma_zeros,
)

# There is no numpy masked array function for radians, so we just multiply thus:
deg2rad = radians(1.0)

class AccelerationLateralSmoothed(DerivedParameterNode):
    '''
    Apply a moving average for two seconds (9 samples) to smooth out spikes
    caused by uneven surfaces - especially noticable during cornering.
    '''

    units = ut.G

    def derive(self, acc=P('Acceleration Lateral')):

        self.window = acc.hz * 2 + 1  # store for ease of testing
        self.array = moving_average(acc.array, window=self.window)

class AltitudeAAL(DerivedParameterNode):
    '''
    This is the main altitude measure used during flight analysis.

    Where radio altimeter data is available, this is used for altitudes up to
    100ft and thereafter the pressure altitude signal is used. The two are
    "joined" together at the sample above 100ft in the climb or descent as
    appropriate.

    If no radio altitude signal is available, the simple measure based on
    pressure altitude only is used, which provides workable solutions except
    that the point of takeoff and landing may be inaccurate.

    This parameter includes a rejection of bounced landings of less than 35ft
    height.
    '''

    name = 'Altitude AAL'
    align_frequency = 2
    align_offset = 0
    units = ut.FT

    @classmethod
    def can_operate(cls, available):
        required = all_of(('Fast', 'Altitude STD Smoothed'), available)
        return required

    def find_liftoff_start(self, alt_std):
        # Test case => NAX_8_LN-NOE_20120109063858_02_L3UQAR___dev__sdb.002.hdf5
        # Look over the first 500ft of climb (or less if the data doesn't get that high).
        first_val = first_valid_sample(alt_std).value
        to = index_at_value(alt_std, min(first_val+500, np.ma.max(alt_std)))
        if to is not None:
            to = int(to)
        # Seek the point where the altitude first curves upwards.
        first_curve = int(peak_curvature(repair_mask(alt_std[:to]),
                                         curve_sense='Concave',
                                         gap = 7,
                                         ttp = 10))

        # or where the rate of climb is > 20ft per second?
        climbing = rate_of_change_array(alt_std, self.frequency)
        climbing[climbing<20] = np.ma.masked
        idx = min(first_curve, first_valid_sample(climbing[:to]).index)
        return idx

    def shift_alt_std(self, alt_std, land_pitch):
        '''
        Return Altitude STD Smoothed shifted relative to 0 for cases where we do not
        have a reliable Altitude Radio.
        '''
        pit = 0.0
        if land_pitch is None or not np.ma.count(land_pitch):
            # This is a takeoff case where we ideally recognise the reduction
            # in pressure at liftoff as the aircraft rotates and the static
            # pressure field around the aircraft changes.
            try:
                idx = self.find_liftoff_start(alt_std)

                # The liftoff most probably arose in the preceding 10
                # seconds. Allow 3 seconds afterwards for luck.
                rotate = slice(max(idx-10*self.frequency,0),
                               idx+3*self.frequency)
                # Draw a straight line across this period with a ruler.
                p,m,c = coreg(alt_std[rotate])
                ruler = np.ma.arange(rotate.stop-rotate.start)*m+c
                # Measure how far the altitude is below the ruler.
                delta = alt_std[rotate] - ruler
                # The liftoff occurs where the gap is biggest because this is
                # where the wing lift has caused the local pressure to
                # increase, hence the altitude appears to decrease.
                pit = alt_std[np.ma.argmin(delta)+rotate.start]

                '''
                # Quick visual check of the operation of the takeoff point detection.
                import matplotlib.pyplot as plt
                plt.plot(alt_std)
                xnew = np.linspace(rotate.start,rotate.stop,num=2)
                ynew = (xnew-rotate.start)*m + c
                plt.plot(xnew,ynew,'-')
                plt.plot(np.ma.argmin(delta)+rotate.start, pit, 'dg')
                plt.plot(idx, alt_std[idx], 'dr')
                plt.show()
                plt.clf()
                plt.close()
                '''

            except:
                # If something odd about the data causes a problem with this
                # technique, use a simpler solution. This can give
                # significantly erroneous results in the case of sloping
                # runways, but it's the most robust technique.
                pit = np.ma.min(alt_std)

        else:

            # This is a landing case where we use the pitch attitude to
            # identify the touchdown point.

            # First find the lowest point
            lowest_index = np.ma.argmin(alt_std)
            lowest_height = alt_std[lowest_index]
            # and go up 50ft
            still_airborne = index_at_value(alt_std[lowest_index:],
                                            lowest_height + 50.0,
                                            endpoint='closing')
            check_slice = slices_int(lowest_index, lowest_index + still_airborne)
            # What was the maximum pitch attitude reached in the last 50ft of the descent?
            max_pitch = max(land_pitch[check_slice])
            # and the last index at this attitude is given by:
            if max_pitch:
                max_pch_idx = (land_pitch[check_slice] == max_pitch).nonzero()[-1][0]
                pit = alt_std[lowest_index + max_pch_idx]

            '''
            # Quick visual check of the operation of the takeoff point detection.
            import matplotlib.pyplot as plt
            show_slice = slice(0, lowest_index + still_airborne)
            plt.plot(alt_std[show_slice] - pit)
            plt.plot(land_pitch[show_slice]*10.0)
            plt.plot(lowest_index + max_pch_idx, 0.0, 'dg')
            plt.show()
            plt.close()
            '''

        return np.ma.maximum(alt_std - pit, 0.0)

    def compute_aal(self, mode, alt_std, low_hb, high_gnd, alt_rad=None,
                    land_pitch=None):

        alt_result = np_ma_zeros_like(alt_std)
        if alt_rad is None or np.ma.count(alt_rad)==0:
            # This backstop trap for negative values is necessary as aircraft
            # without rad alts will indicate negative altitudes as they land.
            if mode != 'land':
                return alt_std - high_gnd
            else:
                return self.shift_alt_std(alt_std, land_pitch)

        if mode=='over_gnd' and (low_hb-high_gnd)>100.0:
            return alt_std - high_gnd

        if mode != 'air':
            # We pretend the aircraft can't go below ground level for altitude AAL:
            alt_rad_aal = np.ma.maximum(alt_rad, 0.0)
            ralt_sections = np.ma.clump_unmasked(
                np.ma.masked_outside(alt_rad_aal, 0.1, ALTITUDE_AAL_TRANS_ALT))
            if len(ralt_sections) == 0:
                # No useful radio altitude signals, so just use pressure altitude
                # and pitch data.
                return self.shift_alt_std(alt_std, land_pitch)

        if mode == 'land':
            # We refine our definition of the radio altimeter sections to
            # take account of bounced landings and altimeters which read
            # small positive values on the ground.
            bounce_sections = [y for y in ralt_sections if np.ma.max(alt_rad[y]) > BOUNCED_LANDING_THRESHOLD]
            if bounce_sections:
                bounce_end = bounce_sections[0].start
                hundred_feet = bounce_sections[-1].stop

                alt_result[bounce_end:hundred_feet] = alt_rad_aal[bounce_end:hundred_feet]
                alt_result[:bounce_end] = 0.0
                ralt_sections = [slice(0, hundred_feet)]

        elif mode=='over_gnd':

            ralt_sections = np.ma.clump_unmasked(
                np.ma.masked_outside(alt_rad_aal, 0.0, ALTITUDE_AAL_TRANS_ALT))
            if len(ralt_sections)==0:
                # Altitude Radio did not drop below ALTITUDE_AAL_TRANS_ALT,
                # so we are better off working with just the pressure altitude signal.
                return self.shift_alt_std(alt_std, land_pitch)

        elif mode=='air':
            return alt_std

        baro_sections = slices_not(ralt_sections, begin_at=0,
                                   end_at=len(alt_std))

        for ralt_section in ralt_sections:
            # Note: below check commented out as multiple commercial airports
            # have an elevation of more than 10000ft. If the use case for
            # this check is dicovered it needs to be documented - 2015/03/20

            #if np.ma.mean(alt_std[ralt_section] -
            # alt_rad_aal[ralt_section]) > 10000:
                ## Difference between Altitude STD and Altitude Radio should not
                ## be greater than 10000 ft when Altitude Radio is recording below
                ## 100 ft. This will not fix cases when Altitude Radio records
                ## spurious data at lower altitudes.
                # continue

            if mode=='over_gnd':
                # land mode is handled above so just need to set rad alt as
                # aal for over ground sections
                alt_result[ralt_section] = alt_rad_aal[ralt_section]

            for baro_section in baro_sections:
                # I know there must be a better way to code these symmetrical processes, but this works :o)
                link_baro_rad_fwd(baro_section, ralt_section, alt_rad_aal, alt_std, alt_result)
                link_baro_rad_rev(baro_section, ralt_section, alt_rad_aal, alt_std, alt_result)

        return alt_result

    def derive(self, alt_rad=P('Altitude Radio'),
               alt_std=P('Altitude STD Smoothed'),
               speedies=S('Fast'),
               pitch=P('Pitch'),
               gog=P('Gear On Ground')):
        # Altitude Radio taken as the prime reference to ensure the minimum
        # ground clearance passing peaks is accurately reflected. Alt AAL
        # forced to 2htz

        # alt_aal will be zero on the airfield, so initialise to zero.
        alt_aal = np_ma_zeros_like(alt_std.array)

        for speedy in speedies:
            quick = speedy.slice
            if quick == slice(None, None, None):
                self.array = alt_aal
                return

            # We set the minimum height for detecting flights to 500 ft. This
            # ensures that low altitude "hops" are still treated as complete
            # flights while more complex flights are processed as climbs and
            # descents of 500 ft or more.
            alt_idxs, alt_vals = cycle_finder(alt_std.array[quick],
                                              min_step=500)

            # Reference to start of arrays for simplicity hereafter.
            if alt_idxs is None:
                continue

            alt_idxs += quick.start or 0

            n = 0
            dips = []
            # List of dicts, with each sublist containing:

            # 'type' of item 'land' or 'over_gnd' or 'high'

            # 'slice' for this part of the data
            # if 'type' is 'land' the land section comes at the beginning of the
            # slice (i.e. takeoff slices are normal, landing slices are
            # reversed)
            # 'over_gnd' or 'air' are normal slices.

            # 'alt_std' as:
            # 'land' = the pressure altitude on the ground
            # 'over_gnd' = the pressure altitude when flying closest to the
            #              ground
            # 'air' = the lowest pressure altitude in this slice

            # 'highest_ground' in this area
            # 'land' = the pressure altitude on the ground
            # 'over_gnd' = the pressure altitude minus the radio altitude when
            #              flying closest to the ground
            # 'air' = None (the aircraft was too high for the radio altimeter to
            #         register valid data

            n_vals = len(alt_vals)
            while n < n_vals - 1:
                alt = alt_vals[n]
                alt_idx = alt_idxs[n]
                next_alt = alt_vals[n + 1]
                next_alt_idx = alt_idxs[n + 1]

                if next_alt > alt:
                    # Rising section.
                    dips.append({
                        'type': 'land',
                        'slice': slice(quick.start, next_alt_idx),
                        # was 'slice': slice(alt_idx, next_alt_idx),
                        'alt_std': alt,
                        'highest_ground': alt,
                    })
                    n += 1
                    continue

                if n + 2 >= n_vals:
                    # Falling section. Slice it backwards to use the same code
                    # as for takeoffs.
                    dips.append({
                        'type': 'land',
                        'slice': slice(quick.stop, (alt_idx or 1) - 1, -1),
                        # was 'slice': slice(next_alt_idx - 1, alt_idx - 1, -1),
                        'alt_std': next_alt,
                        'highest_ground': next_alt,
                    })
                    n += 1
                    continue

                if alt_vals[n + 2] > next_alt:
                    # A down and up section.
                    down_up = slice(alt_idx, alt_idxs[n + 2])
                    # Is radio altimeter data both supplied and valid in this
                    # range?
                    if alt_rad and np.ma.count(alt_rad.array[down_up]) > 0:
                        # Let's find the lowest rad alt reading
                        # (this may not be exactly the highest ground, but
                        # it was probably the point of highest concern!)
                        ##arg_hg_max = \
                            ##np.ma.argmin(alt_rad.array[down_up]) + \
                            ##alt_idxs[n]
                        ##hg_max = alt_std.array[arg_hg_max] - \
                            ##alt_rad.array[arg_hg_max]
                        hb_min = np.ma.min(alt_std.array[down_up])
                        hr_min = np.ma.min(alt_rad.array[down_up])
                        hg_max = hb_min - hr_min
                        if hg_max and hg_max < 20000:
                            # The rad alt measured height above a peak...
                            dips.append({
                                'type': 'over_gnd',
                                'slice': down_up,
                                'alt_std': hb_min,
                                'highest_ground': hg_max,
                            })
                        else:
                            dips.append({
                                'type': 'air',
                                'slice': down_up,
                                'alt_std': hb_min,
                                'highest_ground': None,
                            })
                    else:
                        # We have no rad alt data we can use.
                        # TODO: alt_std code needs careful checking.
                        if dips:
                            prev_dip = dips[-1]
                        if dips and prev_dip['type'] == 'high':
                            # Join this dip onto the previous one
                            prev_dip['slice'] = \
                                slice(prev_dip['slice'].start,
                                      alt_idxs[n + 2])
                            prev_dip['alt_std'] = \
                                min(prev_dip['alt_std'],
                                    next_alt)
                        else:
                            dips.append({
                                'type': 'high',
                                'slice': down_up,
                                'alt_std': next_alt,
                                'highest_ground': next_alt,
                            })
                    n += 2
                else:
                    if n + 3 == n_vals:
                        # Final section has a masked peak, so treat as a
                        # falling section from the highest value.
                        dips.append({
                            'type': 'land',
                            'slice': slice(quick.stop, (alt_idx or 1) - 1, -1),
                            # was 'slice': slice(next_alt_idx - 1, alt_idx - 1, -1),
                            'alt_std': next_alt,
                            'highest_ground': next_alt,
                        })
                        n += 1
                        continue
                    else:
                        # Problem as the data should dip, but instead has a peak. Rare case, so just skip past it.
                        n += 1

            for n, dip in enumerate(dips):
                if dip['type'] == 'high':
                    if n == 0:
                        if len(dips) == 1:
                            # Arbitrary offset in indeterminate case.
                            dip['alt_std'] = dip['highest_ground']+1000.0
                        else:
                            next_dip = dips[n + 1]
                            dip['highest_ground'] = \
                                dip['alt_std'] - next_dip['alt_std'] + \
                                next_dip['highest_ground']
                    elif n == len(dips) - 1:
                        prev_dip = dips[n - 1]
                        dip['highest_ground'] = \
                            dip['alt_std'] - prev_dip['alt_std'] + \
                            prev_dip['highest_ground']
                    else:
                        # Here is the most commonly used, and somewhat
                        # arbitrary code. For a dip where no radio
                        # measurement of the ground is available, what height
                        # can you use as the datum? The lowest ground
                        # elevation in the preceding and following sections
                        # is practical, a little optimistic perhaps, but
                        # useable until we find a case otherwise.

                        # This was modified to ensure the minimum height was
                        # 1000ft as we had a case where the lowest dips were
                        # below the takeoff and landing airfields.
                        next_dip = dips[n + 1]
                        prev_dip = dips[n - 1]
                        if ac_type == helicopter and gog and not alt_rad:
                            on_ground = slices_and(runs_of_ones(gog.array == 'Ground'), [dip['slice']])
                            if on_ground:
                                on_ground = max(on_ground, key=lambda p: p.stop-p.start)
                                dip['highest_ground'] = np.ma.median(alt_std.array[on_ground])
                            else:
                                dip['highest_ground'] = min(prev_dip['highest_ground'],
                                                        next_dip['highest_ground'])
                        else:
                            dip['highest_ground'] = min(prev_dip['highest_ground'],
                                                        dip['alt_std']-1000.0,
                                                        next_dip['highest_ground'])

            for dip in dips:
                if alt_rad and np.ma.count(alt_rad.array[dip['slice']]):
                    alt_rad_section = repair_mask(alt_rad.array[dip['slice']])
                else:
                    alt_rad_section = None

                if (dip['type']=='land') and (alt_rad_section is None) and \
                   (dip['slice'].stop<dip['slice'].start) and pitch:
                    land_pitch=pitch.array[dip['slice']]
                else:
                    land_pitch=None

                alt_aal[dip['slice']] = self.compute_aal(
                    dip['type'],
                    alt_std.array[dip['slice']],
                    dip['alt_std'],
                    dip['highest_ground'],
                    alt_rad=alt_rad_section,
                    land_pitch=land_pitch)

            # Reset end sections only if Altitude STD is not masked for these sections
            if len(alt_idxs):
                if quick.start != alt_idxs[0]+1 and np.ma.count(alt_std.array[quick.start:alt_idxs[0]+1]) > (alt_idxs[0]+1 - quick.start)/2:
                    alt_aal[quick.start:alt_idxs[0]+1] = 0.0
                if alt_idxs[-1]+1 != quick.stop and np.ma.count(alt_std.array[alt_idxs[-1]+1:quick.stop]) > (quick.stop - alt_idxs[-1]+1)/2:
                    alt_aal[alt_idxs[-1]+1:quick.stop] = 0.0

        self.array = alt_aal


def link_baro_rad_fwd(baro_section, ralt_section, alt_rad, alt_std, alt_result):
    begin_index = baro_section.start

    if ralt_section.stop == begin_index:
        start_plus_60 = min(begin_index + 60, len(alt_std))
        alt_diff = (alt_std[begin_index:start_plus_60] -
                    alt_rad[begin_index:start_plus_60])
        slip, up_diff = first_valid_sample(alt_diff)
        if slip is None:
            up_diff = 0.0
        else:
            # alt_std is invalid at the point of handover
            # so stretch the radio signal until we can
            # handover.
            fix_slice = slice(begin_index,
                              begin_index + slip)
            alt_result[fix_slice] = alt_rad[fix_slice]
            begin_index += slip

        alt_result[begin_index:] = \
            alt_std[begin_index:] - up_diff

def link_baro_rad_rev(baro_section, ralt_section, alt_rad, alt_std, alt_result):
    end_index = baro_section.stop

    if ralt_section.start == end_index:
        end_minus_60 = max(end_index-60, 0)
        alt_diff = (alt_std[end_minus_60:end_index] -
                    alt_rad[end_minus_60:end_index])
        slip, up_diff = first_valid_sample(alt_diff[::-1])
        if slip is None:
            up_diff = 0.0
        else:
            # alt_std is invalid at the point of handover
            # so stretch the radio signal until we can
            # handover.
            fix_slice = slice(end_index-slip,
                              end_index)
            alt_result[fix_slice] = alt_rad[fix_slice]
            end_index -= slip

        alt_result[:end_index] = \
            alt_std[:end_index] - up_diff


class AltitudeAALForFlightPhases(DerivedParameterNode):
    '''
    This parameter repairs short periods of masked data, making it suitable for
    detecting altitude bands on the climb and descent. The parameter should not
    be used to compute KPV values themselves, to avoid using interpolated
    values in an event.
    '''

    name = 'Altitude AAL For Flight Phases'
    units = ut.FT

    def derive(self, alt_aal=P('Altitude AAL')):

        self.array = np.ma.maximum(repair_mask(alt_aal.array, repair_duration=None),
                                   0.0)

class AltitudeSTDSmoothed(DerivedParameterNode):
    """
    This applies various smoothing functions depending upon the quality of the source data, then
    in all cases applies a local average smoothing. In particular, this ensures that the derived
    Vertical Speed parameter matches the response seen by the pilot and is not excesively affected
    by turbulence.
    """

    '''
    :param frame: The frame attribute, e.g. '737-i'
    :type frame: An attribute

    :returns Altitude STD Smoothed as a local average where the original source is unacceptable, but unchanged otherwise.
    :type parameter object.
    '''

    name = 'Altitude STD Smoothed'
    align = False
    units = ut.FT

    @classmethod
    def can_operate(cls, available):

        return ('Frame' in available and
                ('Altitude STD' in available))

    def derive(self, alt=P('Altitude STD')):
        self.array = alt.array
        # Applying moving_window of a moving_window to avoid a large weighting/
        # window size which would skew sharp curves.
        self.array = moving_average(moving_average(self.array))


class BaroCorrection(DerivedParameterNode):
    '''This computes the Baro correction by either merging
    Baro Correction (Capt) and (FO) or by using the difference
    between Altitude Baro and Altitude STD.
    '''

    units = ut.MILLIBAR

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude STD', 'Altitude Baro'), available)

    def derive(self,
               alt_baro=P('Altitude Baro'),
               alt_std=P('Altitude STD')):

            baro = alt2press(alt_std.array - alt_baro.array)
            self.array = baro.array

# TODO: This would be cool to have, but maybe at a later stage
# class AltitudeTail(DerivedParameterNode):
#     """
#     This derived parameter computes the height of the tail above the runway,
#     as a measure of the likelyhood of a tailscrape.

#     We are only interested in the takeoff, go-around and landing phases,
#     where the tail may be close to scraping the runway. For this reason, we
#     don't use pressure altitmetry as this suffers too badly from pressure
#     variations at the point of liftoff and touchdown.

#     The parameter dist_gear_to_tail is measured in metres and is the
#     horizontal distance aft from the main gear to the point on the tail most
#     likely to scrape the runway.

#     Parameter ground_to_tail is the height of the point most likely to
#     scrape. Ideally this is computed from the manufacturer-provided
#     tailscrape attitude at the point of liftoff:

#     ground_to_tail = dist_gear_to_tail*tan(tailscrape attitude at liftoff)
#     """

#     units = ut.FT

#     def derive(self, alt_rad=P('Altitude Radio'), pitch=P('Pitch'),
#                toffs=S('Takeoff'), gas=S('Go Around And Climbout'),
#                lands=S('Landing'),
#                ground_to_tail=A('Ground To Lowest Point Of Tail'),
#                dist_gear_to_tail=A('Main Gear To Lowest Point Of Tail')):

#         # Collect the periods we are interested in:
#         phases=slices_or([t.slice for t in toffs],
#                          [g.slice for g in gas],
#                          [l.slice for l in lands],
#                          )

#         # Scale the aircraft geometry into feet to match aircraft altimetry.
#         gear2tail = ut.convert(dist_gear_to_tail.value, ut.METER, ut.FT)
#         ground2tail = ut.convert(ground_to_tail.value, ut.METER, ut.FT)

#         result = np_ma_masked_zeros_like(alt_rad.array)
#         # The tail clearance is the value with the aircraft settled on its
#         # wheels plus radio atimetry minus the pitch attitude change at that
#         # tail arm.
#         for phase in phases:
#             result[phase] = alt_rad.array[phase] + ground2tail - \
#                 np.ma.tan(pitch.array[phase]*deg2rad) * gear2tail
#         self.array = result


##############################################################################
# Automated Systems


class CabinAltitude(DerivedParameterNode):
    '''
    Some aircraft record the cabin altitude in feet, while others record the
    cabin pressure (normally in psi). This function converts the pressure
    reading to altitude equivalent, so that the KPVs can operate only in
    altitude units. After all, the crew set the cabin altitude, not the
    pressure.

    Typically aircraft also have the 'Cabin Altitude Warning' discrete parameter.
    '''

    units = ut.FT

    def derive(self, cp=P('Cabin Press')):

        # XXX: assert cp.units == ut.PSI  # Would like to assert units as 'psi'
        self.array = press2alt(cp.array)


class ClimbForFlightPhases(DerivedParameterNode):
    '''
    This computes climb segments, and resets to zero as soon as the aircraft
    descends. Very useful for measuring climb after an aborted approach etc.
    '''

    units = ut.FT

    def derive(self, alt_std=P('Altitude STD Smoothed'), airs=S('Fast')):

        self.array = np_ma_zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            ups = np.ma.clump_unmasked(np.ma.masked_less(deltas,0.0))
            for up in ups:
                self.array[air.slice][up] = np.ma.cumsum(deltas[up])


class DescendForFlightPhases(DerivedParameterNode):
    '''
    This computes descent segments, and resets to zero as soon as the aircraft
    climbs Used for measuring descents, e.g. following a suspected level bust.
    '''

    units = ut.FT

    def derive(self, alt_std=P('Altitude STD Smoothed'), airs=S('Fast')):

        self.array = np_ma_zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            downs = np.ma.clump_unmasked(np.ma.masked_greater(deltas,0.0))
            for down in downs:
                self.array[air.slice][down] = np.ma.cumsum(deltas[down])


class AOA(DerivedParameterNode):
    '''
    Angle of Attack - averages Left and Right signals to account for side slip.
    See Bombardier AOM-1281 document for further details.
    '''

    name = 'AOA'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('AOA (L)', 'AOA (R)'), available)

    def derive(self, aoa_l=P('AOA (L)'), aoa_r=P('AOA (R)')):

        if aoa_l and aoa_r:
            # Convert the actual values as simconnect is ridiculous
            aoa_l_converted = 180 - aoa_l.array
            aoa_r_converted = 180 - aoa_r.array
            # Average angle of attack to compensate for sideslip.
            self.array = (aoa_l_converted + aoa_r_converted) / 2
        else:
            # only one available
            aoa = aoa_l or aoa_r
            # Convert the actual values
            aoa_converted = 180 - aoa.array
            self.array = aoa_converted


class ControlInputAngle(DerivedParameterNode):
    '''
    Angle of the captain's side stick.

    This parameter calcuates the combined angle from the separate pitch and
    roll component angles of the sidestick for the captain.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    name = 'Control Input Angle'
    units = ut.PERCENT

    def derive(self,
               pitch_capt=M('Control Input Pitch'),
               roll_capt=M('Control Input Roll')):

        pitch_repaired = repair_mask(pitch_capt.array, pitch_capt.frequency)
        roll_repaired = repair_mask(roll_capt.array, roll_capt.frequency)
        self.array = np.ma.sqrt(pitch_repaired ** 2 + roll_repaired ** 2)
        

class DistanceToLanding(DerivedParameterNode):
    '''
    Ground distance to cover before touchdown.

    Note: This parameter gets closer to zero approaching each touchdown,
    but then increases as the aircraft decelerates on the runway after the
    final touchdown.

    This does not reflect go-arounds at the present time.
    '''

    units = ut.NM

    def derive(self, dist=P('Distance Travelled'), tdwns=KTI('Touchdown')):
        self.array = np.zeros_like(dist.array)
        if tdwns:
            last_tdwn = 0
            for tdwn in tdwns.get_ordered_by_index():
                this_tdwn = int(tdwn.index)
                self.array[last_tdwn:this_tdwn+1] = np.ma.abs(dist.array[last_tdwn:this_tdwn+1] - (value_at_index(dist.array, this_tdwn) or np.ma.masked))
                last_tdwn = this_tdwn+1
            self.array[last_tdwn:] = np.ma.abs(dist.array[last_tdwn:] - dist.array[this_tdwn])
        else:
            self.array.mask = True


class DistanceFlown(DerivedParameterNode):
    '''
    Distance flown in Nautical Miles. Calculated using integral of
    Airspeed True.
    '''

    units = ut.NM

    def derive(self, tas=P('Airspeed True'), airs=S('Airborne')):

        self.array = np_ma_zeros_like(tas.array)
        if airs.get_first():
            start = airs.get_first().slice.start
            stop = airs.get_last().slice.stop
            _slice = slice(start, stop)
            self.array[_slice] = integrate(tas.array[_slice], tas.frequency, scale=1.0 / 3600.0)
            self.array[_slice.stop:] = self.array[_slice.stop-1]


class DistanceTravelled(DerivedParameterNode):
    '''
    Distance travelled in Nautical Miles. Calculated using integral of
    Groundspeed.
    '''

    units = ut.NM

    def derive(self, gspd=P('Groundspeed')):
        gspdarray = repair_mask(gspd.array, gspd.frequency,
                                repair_duration=None)
        self.array = integrate(gspdarray, gspd.frequency, scale=1.0 / 3600.0)


class Drift(DerivedParameterNode):
    '''
    Drift angle in degrees. Calculated using the difference between
    Track and Heading.
    '''
    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return all_of(('Heading Continuous', 'Track'), available)

    def derive(self,
               track=P('Track'),
               heading=P('Heading Continuous')):

        self.frequency = track.frequency
        self.offset = track.offset
        self.array = (track.array - align(heading, track) + 180) % 360 - 180

##############################################################################
# Engine EPR


class Eng_EPRAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Avg'
    align = False
    units = None

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_EPRMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Max'
    align = False
    units = None

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_EPRMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Min'
    align = False
    units = None

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_EPRMinFor5Sec(DerivedParameterNode):
    '''
    Returns the lowest EPR for up to four engines over five seconds.
    '''

    name = 'Eng (*) EPR Min For 5 Sec'
    align_frequency = 2
    align_offset = 0
    units = None

    def derive(self, eng_epr_min=P('Eng (*) EPR Min')):
        self.array = second_window(eng_epr_min.array, self.frequency, 5)


class Eng_EPRAvgFor10Sec(DerivedParameterNode):
    '''
    Returns the average EPR for up to four engines over 10 seconds.
    '''

    name = 'Eng (*) EPR Avg For 10 Sec'
    align_frequency = 1  # force odd freq for 10 sec window
    units = None

    def derive(self, eng_epr_avg=P('Eng (*) EPR Avg')):
        self.array = second_window(eng_epr_avg.array, self.frequency, 10)

##############################################################################
# Engine Fuel Flow


class Eng_FuelFlow(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Fuel Flow'
    align = False
    units = ut.LBS_H

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Fuel Flow'),
               eng2=P('Eng (2) Fuel Flow'),
               eng3=P('Eng (3) Fuel Flow'),
               eng4=P('Eng (4) Fuel Flow')):

        # assume all engines Fuel Flow are record at the same frequency
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.sum(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_FuelFlowMin(DerivedParameterNode):
    '''
    The minimum recorded Fuel Flow across all engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) Fuel Flow Min'
    align_frequency = 4
    align_offset = 0
    units = ut.KG_H

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Fuel Flow'),
               eng2=P('Eng (2) Fuel Flow'),
               eng3=P('Eng (3) Fuel Flow'),
               eng4=P('Eng (4) Fuel Flow')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_FuelFlowMax(DerivedParameterNode):
    '''
    The maximum recorded Fuel Flow across all engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) Fuel Flow Max'
    align_frequency = 4
    align_offset = 0
    units = ut.KG_H

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Fuel Flow'),
               eng2=P('Eng (2) Fuel Flow'),
               eng3=P('Eng (3) Fuel Flow'),
               eng4=P('Eng (4) Fuel Flow')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


##############################################################################
# Fuel Burn


class Eng_1_FuelBurn(DerivedParameterNode):
    '''
    Amount of fuel burnt since the start of the data.
    '''

    name = 'Eng (1) Fuel Burn'
    units = ut.KG

    def derive(self, ff=P('Eng (1) Fuel Flow')):

        flow = repair_mask(ff.array)
        flow = np.ma.where(flow.mask, 0.0, flow)
        self.array = np.ma.array(integrate(flow / 3600.0, ff.frequency))


class Eng_2_FuelBurn(DerivedParameterNode):
    ''''
    Amount of fuel burnt since the start of the data.
    '''

    name = 'Eng (2) Fuel Burn'
    units = ut.KG

    def derive(self, ff=P('Eng (2) Fuel Flow')):

        flow = repair_mask(ff.array)
        flow = np.ma.where(flow.mask, 0.0, flow)
        self.array = np.ma.array(integrate(flow / 3600.0, ff.frequency))


class Eng_3_FuelBurn(DerivedParameterNode):
    '''
    Amount of fuel burnt since the start of the data.
    '''

    name = 'Eng (3) Fuel Burn'
    units = ut.KG

    def derive(self, ff=P('Eng (3) Fuel Flow')):

        flow = repair_mask(ff.array)
        flow = np.ma.where(flow.mask, 0.0, flow)
        self.array = np.ma.array(integrate(flow / 3600.0, ff.frequency))


class Eng_4_FuelBurn(DerivedParameterNode):
    '''
    Amount of fuel burnt since the start of the data.
    '''

    name = 'Eng (4) Fuel Burn'
    units = ut.KG

    def derive(self, ff=P('Eng (4) Fuel Flow')):

        flow = repair_mask(ff.array)
        flow = np.ma.where(flow.mask, 0.0, flow)
        self.array = np.ma.array(integrate(flow / 3600.0, ff.frequency))


class Eng_FuelBurn(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Fuel Burn'
    align = False
    units = ut.KG

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Fuel Burn'),
               eng2=P('Eng (2) Fuel Burn'),
               eng3=P('Eng (3) Fuel Burn'),
               eng4=P('Eng (4) Fuel Burn')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.sum(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


##############################################################################
# Engine Gas Temperature


class Eng_GasTempAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Avg'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_GasTempMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Max'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_GasTempMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Min'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


##############################################################################
# Engine N1

# TODO: This parameter is stupid because simconnect records RPM and not % and it will be very hard to get reliable 100% rpm values for engines, but maybe one day?        
class Eng1N1(DerivedParameterNode):
    '''
    This converts Eng (1) N1 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (1) N1'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (1) N1 RPM' in available)
    
    def derive(self, n1=P('Eng (1) N1 RPM')):
        self.array = (n1.array - np.min(n1.array)) / (np.max(n1.array) - np.min(n1.array)) * 100
    
class Eng2N1(DerivedParameterNode):
    '''
    This converts Eng (2) N1 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (2) N1'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (2) N1 RPM' in available)
    
    def derive(self, n1=P('Eng (2) N1 RPM')):
        self.array = (n1.array - np.min(n1.array)) / (np.max(n1.array) - np.min(n1.array)) * 100


class Eng3N1(DerivedParameterNode):
    '''
    This converts Eng (3) N1 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (3) N1'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (3) N1 RPM' in available)
    
    def derive(self, n1=P('Eng (3) N1 RPM')):
        self.array = (n1.array - np.min(n1.array)) / (np.max(n1.array) - np.min(n1.array)) * 100


class Eng4N1(DerivedParameterNode):
    '''
    This converts Eng (4) N1 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (4) N1'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (4) N1 RPM' in available)
    
    def derive(self, n1=P('Eng (4) N1 RPM')):
        self.array = (n1.array - np.min(n1.array)) / (np.max(n1.array) - np.min(n1.array)) * 100

class Eng_N1Avg(DerivedParameterNode):
    '''
    This returns the avaerage N1 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N1 Avg'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
           eng1=P('Eng (1) N1'),
           eng2=P('Eng (2) N1'),
           eng3=P('Eng (3) N1'),
           eng4=P('Eng (4) N1')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])

class Eng_N1AvgFor10Sec(DerivedParameterNode):
    '''
    Returns the average N1 for up to four engines over 10 seconds.
    '''

    name = 'Eng (*) N1 Avg For 10 Sec'
    align_frequency = 1  # force odd freq for 10 sec window
    units = ut.PERCENT

    def derive(self, eng_n1_avg=P('Eng (*) N1 Avg')):
        self.array = second_window(eng_n1_avg.array, self.frequency, 10)


class Eng_N1Max(DerivedParameterNode):
    '''
    This returns the highest N1 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N1 Max'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_N1Min(DerivedParameterNode):
    '''
    This returns the lowest N1 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N1 Min'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_N1Split(DerivedParameterNode):
    '''
    Simple detection of largest engine speed mismatch.
    '''

    name = 'Eng (*) N1 Split'
    align_frequency = 1
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):
        return all_of(cls.get_dependency_names(), available)

    def derive(self,
               n1max=P('Eng (*) N1 Max'),
               n1min=P('Eng (*) N1 Min')):

        '''
        Clinky way of making sure the masked arrays hold zeros, just to make debugging easier.
        '''
        zeros = np_ma_masked_zeros_like(n1max.array)
        diff = n1max.array - n1min.array
        self.array = np.ma.where(diff.mask, zeros, diff)


class Eng_N1MinFor5Sec(DerivedParameterNode):
    '''
    Returns the lowest N1 for up to four engines over five seconds.
    '''

    name = 'Eng (*) N1 Min For 5 Sec'
    align_frequency = 2
    align_offset = 0
    units = ut.PERCENT

    def derive(self, eng_n1_min=P('Eng (*) N1 Min')):
        self.array = second_window(eng_n1_min.array, self.frequency, 5, extend_window=True)


##############################################################################
# Engine N2

class Eng1N2(DerivedParameterNode):
    '''
    This converts Eng (1) N2 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (1) N2'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (1) N2 RPM' in available)
    
    def derive(self, n2=P('Eng (1) N2 RPM')):
        self.array = (n2.array - np.min(n2.array)) / (np.max(n2.array) - np.min(n2.array)) * 100


class Eng2N2(DerivedParameterNode):
    '''
    This converts Eng (2) N2 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (2) N2'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (2) N2 RPM' in available)
    
    def derive(self, n2=P('Eng (2) N2 RPM')):
        self.array = (n2.array - np.min(n2.array)) / (np.max(n2.array) - np.min(n2.array)) * 100


class Eng3N2(DerivedParameterNode):
    '''
    This converts Eng (3) N2 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (3) N2'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (3) N2 RPM' in available)
    
    def derive(self, n2=P('Eng (3) N2 RPM')):
        self.array = (n2.array - np.min(n2.array)) / (np.max(n2.array) - np.min(n2.array)) * 100


class Eng4N2(DerivedParameterNode):
    '''
    This converts Eng (4) N2 RPM to a percentage by taking the max value and treating it as 100%, and min value as 0%
    '''
    
    name = 'Eng (4) N2'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return super().can_operate(available) and ('Eng (4) N2 RPM' in available)
    
    def derive(self, n2=P('Eng (4) N2 RPM')):
        self.array = (n2.array - np.min(n2.array)) / (np.max(n2.array) - np.min(n2.array)) * 100
        
class Eng_N2Avg(DerivedParameterNode):
    '''
    This returns the avaerage N2 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N2 Avg'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_N2Max(DerivedParameterNode):
    '''
    This returns the highest N2 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N2 Max'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_N2Min(DerivedParameterNode):
    '''
    This returns the lowest N2 in any sample period for up to four engines.

    All engines data aligned (using interpolation) and forced the frequency to
    be a higher 4Hz to protect against smoothing of peaks.
    '''

    name = 'Eng (*) N2 Min'
    align_frequency = 4
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        

##############################################################################
# Engine Oil Pressure


class Eng_OilPressAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Avg'
    align = False
    units = ut.PSI

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_OilPressMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Max'
    align = False
    units = ut.PSI

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_OilPressMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Min'
    align = False
    units = ut.PSI

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


##############################################################################
# Engine Oil Quantity


class Eng_OilQtyAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Avg'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_OilQtyMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Max'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_OilQtyMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Min'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


##############################################################################
# Engine Oil Temperature


class Eng_OilTempAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Avg'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        avg_array = np.ma.average(engines, axis=0)
        if np.ma.count(avg_array) != 0:
            self.array = avg_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(avg_array)


class Eng_OilTempMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Max'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        max_array = np.ma.max(engines, axis=0)
        if np.ma.count(max_array) != 0:
            self.array = max_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(max_array)


class Eng_OilTempMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Min'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        min_array = np.ma.min(engines, axis=0)
        if np.ma.count(min_array) != 0:
            self.array = min_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(min_array)


##############################################################################
# Engine Torque


class Eng_TorqueAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Avg'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_TorqueMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Max'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_TorqueMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Min'
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):

        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class TorqueAsymmetry(DerivedParameterNode):
    '''
    '''

    align_frequency = 1 # Forced alignment to allow fixed window period.
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion'), ac_type=A('Aircraft Type')):
        turbo_prop = eng_type and eng_type.value == 'PROP'
        required = ['Eng (*) Torque Max', 'Eng (*) Torque Min']
        return (ac_type == helicopter or turbo_prop) and all_of(required, available)

    def derive(self, torq_max=P('Eng (*) Torque Max'), torq_min=P('Eng (*) Torque Min')):
        diff = (torq_max.array - torq_min.array)
        window = 5 # 5 second window
        self.array = moving_average(diff, window=window)


##############################################################################
# Engine Vibration (N1)


class Eng_VibN1Max(DerivedParameterNode):
    '''
    This derived parameter condenses all the available first shaft order
    vibration measurements into a single consolidated value.
    '''

    name = 'Eng (*) Vib N1 Max'
    align = False
    units = None

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Vib N1'),
               eng2=P('Eng (2) Vib N1'),
               eng3=P('Eng (3) Vib N1'),
               eng4=P('Eng (4) Vib N1')):

        params = eng1, eng2, eng3, eng4
        engines = vstack_params(*params)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', params)


##############################################################################

class ZeroFuelWeight(DerivedParameterNode):
    '''
    The aircraft zero fuel weight is computed from the recorded gross weight
    and fuel data.

    See also the GrossWeightSmoothed calculation which uses fuel flow data to
    obtain a higher sample rate solution to the aircraft weight calculation,
    with a best fit to the available weight data.

    TODO: Move to a FlightAttribute which is stored in the database.
    '''

    units = ut.KG
    # Force align for cases when only attribute dependencies are available.
    align_frequency = 1
    align_offset = 0

    @classmethod
    def can_operate(cls, available):
        return ('HDF Duration' in available and all_of(('Fuel Qty', 'Gross Weight'), available))

    def derive(self, fuel_qty=P('Fuel Qty'), gross_wgt=P('Gross Weight'), duration=A('HDF Duration')):
        weight = np.ma.median(gross_wgt.array - fuel_qty.array)
        self.array = np.ma.ones(int(duration.value * self.frequency)) * weight


class GroundspeedSigned(DerivedParameterNode):
    '''
    Adds a negative sign to the pushback movement off the gate to improve ground track computations.

    Also checks the taxi groundspeeds against the recorded position data.
    '''

    units = ut.KT

    def derive(self,
               gspd=P('Groundspeed'),
               power=P('Eng (*) Any Running'),
               taxis=S('Taxiing'),
               lat=P('Latitude Prepared'),
               lon=P('Longitude Prepared'),
               ):

        self.array = gspd.array
        # Ignore the pushback, when the aircraft can have a groundspeed
        # recorded, but in effect it's negative.
        no_power = np.ma.clump_masked(np.ma.masked_less(power.array, 1))
        pushbacks = slices_remove_small_slices(no_power)
        if pushbacks:
            # We sometimes see engines started while the aircraft is being pushed back, so
            # we scan forwards for the faster movement forward and back to find the end of
            # the stationary period.
            move_off = index_at_value(gspd.array, 10.0, _slice=slice(pushbacks[0].stop,None))
            end_stationary = index_at_value(gspd.array, 0.0, _slice=slice(move_off, pushbacks[0].stop, -1))
            if end_stationary:
                self.array[slices_int(pushbacks[0].start, end_stationary)]*=(-1.0)
            else:
                self.array[pushbacks[0]]*=(-1.0)

        for taxi in taxis:
            tx = taxi.slice
            gsp = groundspeed_from_position(lat.array[tx], lon.array[tx], lat.frequency)
            self.array[tx] = np.ma.minimum(gspd.array[tx], gsp)


class FlapAngle(DerivedParameterNode):
    '''
    Gather the recorded flap angle parameters and convert into a single
    analogue.

    Interleaves each of the available flap angle signals into one array, uses
    the sampling offsets (parameters do not need to be evenly sampled in the
    frame) to integrate the resulting array at the combined frequency.
    '''

    align = False
    units = ut.DEGREE

    apply_median_filter = True

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Flap Angle (L)', 'Flap Angle (R)',
            'Flap Angle (C)', 'Flap Angle (MCP)',
            'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard',
        ), available)

    def derive(self,
               flap_A=P('Flap Angle (L)'),
               flap_B=P('Flap Angle (R)'),
               flap_C=P('Flap Angle (C)'),
               flap_D=P('Flap Angle (MCP)'),
               flap_A_inboard=P('Flap Angle (L) Inboard'),
               flap_B_inboard=P('Flap Angle (R) Inboard')):
        flap_A = flap_A or flap_A_inboard
        flap_B = flap_B or flap_B_inboard

        sources = [f for f in (flap_A, flap_B, flap_C, flap_D) if f]

        # if only one parameter, align and use that parameter - easy
        if len(sources) == 1:
            self.array = sources[0].array
            self.offset = sources[0].offset
            self.frequency = sources[0].frequency
            return

        if len(sources) == 3:
            # we can only work with 2 or 4 sources to make the math easier on
            # the hz and interpolation
            sources = sources[:2]


        # sort parameters into ascending offsets
        sources = sorted(sources, key=lambda f: f.offset)

        # interleave data sources so that the x axes is in the correct order
        self.hz = sources[0].hz * len(sources)
        self.offset = sources[0].offset
        if self.offset > 1./self.hz:
            # offset appears too late into the data; make it as late as allowed
            self.warning("Flap Angle sources have similar offsets - " \
                         "check the frame to see if the L and R sources are " \
                         "worth merging or are taken from the same sensors.")
            self.offset = 1./self.hz - 0.00001
        base_hz = sources[0].hz
        duration = len(sources[0].array) / float(sources[0].hz)  # duration of flight in seconds

        xx = []
        yy = []
        for flap in sources:
            assert flap.hz == base_hz, "Can only operate with same flap " \
                   "signals at same frequencies (reshape requires same " \
                   "length arrays). We have: %s which should be at the base " \
                   "frequency of %sHz" % (flap, base_hz)
            xaxis = np.arange(duration, step=1/flap.hz) + flap.offset
            xx.append(xaxis)
            # We do not repair flap.array. If multiple sensors, blend_two_parameters
            # will take care of filling the missing values with values from the
            # good sensor.
            yy.append(flap.array)
            ##scatter(xaxis, flap.array, edgecolor='none', c=col) # col was in zip with sources in for loop

        # if all have the same frequency, offsets are a multiple of the
        # values and they complete they are all equally spaced, we don't need
        # to do any interpolation
        ##TODO: Couldn't work out how to do this in a pretty way!


        # else we have an incomplete set of parameters or are unequally
        # spaced, we need to resample the data with linear interpolation
        # between all of the signals to obtain equally spaced data.

        # create new x axis same length in time but with twice the frequency (step)
        new_xaxis = np.arange(duration, step=1/self.hz) + self.offset # check *2

        # rearrange data into order using ravel/reshape
        new_yaxis = interp(new_xaxis,
                           np.vstack(xx).ravel(order='F'),  # numpy array works
                           np.ma.vstack(yy).data.ravel(order='F'),
                           ##np.ma.vstack(yy).reshape(len(flap.array)*2, order='F'),  # masked array doesn't support order argument yet!
                           )
        # apply median filter to remove spikes where interpolating between
        # two similar but different values and convert to masked array
        if self.apply_median_filter:
            self.array = np.ma.array(medfilt(new_yaxis, 5))
        else:
            self.array = np.ma.array(new_yaxis)
        ##scatter(new_xaxis, self.array, edgecolor='none', c='r')

        if len(sources) == 2:
            self.array, self.frequency, self.offset = blend_two_parameters(*sources)


class FlapSynchroAsymmetry(DerivedParameterNode):
    '''
    Flap Synchro Asymmetry angle.

    Shows an absolute value of difference between Left and Right Flap Synchros.
    Note: this is not a difference in flap angle.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return all_of(('Flap Angle (L) Synchro', 'Flap Angle (R) Synchro',), available)

    def derive(self, synchro_l=P('Flap Angle (L) Synchro'), synchro_r=P('Flap Angle (R) Synchro'),):
        self.array = np.abs(synchro_l.array - synchro_r.array)


'''
class SlatAngle(DerivedParameterNode):

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),
'''


class SlatAngle(DerivedParameterNode):
    '''
    Combines Slat Angle (L) and Slat Angle (R) if available alternativly
    looks up appropriate slat angles for discrete slat positions.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if any_of(('Slat Angle (L)', 'Slat Angle (R)'), available):
            return True
        elif 'Slat Angle Recorded' in available:
            return True
        else:
            if not all_of(('Slat Fully Extended', 'Model', 'Series', 'Family'), available):
                return False
            try:
                at.get_slat_map(model.value, series.value, family.value)
            except KeyError:
                cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                          model.value, series.value, family.value)
                return False

            return True

    def derive(self, slat_l=P('Slat Angle (L)'), slat_r=P('Slat Angle (R)'),
               slat_full=M('Slat Fully Extended'), slat_part=M('Slat Part Extended'),
               slat_retracted=M('Slat Retracted'), slat_angle_rec=P('Slat Angle Recorded'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        if slat_l or slat_r:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(slat_l, slat_r)
        elif slat_angle_rec:
            # Spikey Slat Angle parameter renamed to Slat Angle Recorded to
            # allow removal of single spikes
            self.frequency = slat_angle_rec.frequency
            self.offset = slat_angle_rec.offset
            self.array = second_window(slat_angle_rec.array, 1, 2) # 3 sample smoothing
        else:
            detents = sorted(at.get_slat_map(model.value, series.value, family.value).keys())
            # align
            master = first_valid_parameter(slat_full, slat_part, slat_retracted)
            self.frequency = master.frequency
            self.offset = master.offset
            if slat_retracted:
                # If Retracted parameter use it
                array = np_ma_masked_zeros_like(master.array)
                slat_retracted = slat_retracted.get_aligned(master)
                array[slat_retracted.array == 'Retracted'] = detents[0]
            else:
                # If no explicit Retracted parameter default to Retracted
                array = np_ma_zeros_like(master.array)
            if slat_full:
                array[slat_full.array == 'Extended'] = detents[-1]
            if slat_part:
                part_extended = slat_part.get_aligned(master)
                array[part_extended.array == 'Part Extended'] = detents[1]
            # TODO: Handle slat in transit parameter
            self.array = nearest_neighbour_mask_repair(array)


class _SlopeMixin(object):
    '''
    Mixin class to provide similar functionality to Slope To Landing and to
    Slope To Aiming Point.
    '''
    def calculate_slope(self, alt_aal, dist, alt_std, sat, apps):
        '''
        Calculate the slope based on Altitude AAL, the reference distance (could
        be Distance To Landing or Aiming Point Range) and correcting for ISA
        deviation using Altitude STD, SAT and Approach and Landing sections.

        :param alt_aal: Altitude AAL
        :type alt_aal: Parameter object
        :param dist: Distance to a reference point on the runway
        :type dist: Parameter object
        :param alt_std: Altitude STD
        :type alt_std: Parameter object
        :param sat: SAT
        :type sat: Parameter object
        :param apps: Approach and Landing sections
        :type sat: SectionNode
        :returns: Slope to the reference point on the runway
        :rtype: np.ma.array
        '''
        array = np_ma_masked_zeros_like(alt_aal.array)
        for app in apps:
            if not np.ma.count(alt_aal.array[app.slice]):
                continue
            # What's the temperature deviation from ISA at landing?
            land_alt = last_valid_sample(alt_std.array[app.slice]).value
            land_sat = last_valid_sample(sat.array[app.slice]).value
            # alt and sat can both be valid zero values, hence clunky test:
            if land_alt is not None and land_sat is not None:
                dev = from_isa(land_alt, land_sat)
                # now correct the altitude for temperature deviation.
                alt = alt_dev2alt(alt_aal.array[app.slice], dev)
                array[app.slice] = alt / ut.convert(dist.array[app.slice], ut.NM, ut.FT)
        return array


class SlopeToLanding(DerivedParameterNode, _SlopeMixin):
    '''
    This parameter was developed as part of the Artificical Intelligence
    analysis of approach profiles, 'Identifying Abnormalities in Aircraft
    Flight Data and Ranking their Impact on the Flight' by Dr Edward Smart,
    Institute of Industrial Research, University of Portsmouth.
    http://eprints.port.ac.uk/4141/

    Amended July 2017 to allow for changes in SAT from ISA standard.
    '''

    units = None # This is computed as a ratio of distances, so the tangent of the descent path angle.

    def derive(self, alt_aal=P('Altitude AAL'),
               dist=P('Distance To Landing'),
               alt_std=P('Altitude STD'),
               sat=P('SAT'),
               apps=S('Approach')):

        self.array = self.calculate_slope(alt_aal, dist, alt_std, sat, apps)


class SlopeAngleToLanding(DerivedParameterNode):
    '''
    This parameter calculates the slope angle in degrees.
    '''

    units = ut.DEGREE

    def derive(self, slope_to_ldg=P('Slope To Landing')):

        self.array = np.degrees(np.arctan(slope_to_ldg.array))


class SlopeToAimingPoint(DerivedParameterNode, _SlopeMixin):
    '''
    Slope to the Aiming Point.

    Amended June 2019 to allow for changes in SAT from ISA standard.
    '''

    units = None

    def derive(self, alt_aal=P('Altitude AAL'),
               dist=P('Aiming Point Range'),
               alt_std=P('Altitude STD'),
               sat=P('SAT'),
               apps=S('Approach')):

        self.array = self.calculate_slope(alt_aal, dist, alt_std, sat, apps)


class SlopeAngleToAimingPoint(DerivedParameterNode):
    '''
    This parameter calculates the slope angle in degrees.
    '''

    units = ut.DEGREE

    def derive(self, slope_to_ldg=P('Slope To Aiming Point')):

        self.array = np.degrees(np.arctan(slope_to_ldg.array))


class ApproachFlightPathAngle(DerivedParameterNode):
    '''
    This parameter calculates the slope angle (in degrees) of a landing.

    The Altitude AAL is adjusted according to the ISA standard using SAT at landing.
    Parameter is calculated from the start of the approach phase to 200ft to avoid
    spike as the angle rapidly changes when passing over the aiming point.
    At 500-200 Coreg (correlation and linear regression) calculations is used
    to further straighten the path to a point between the aiming point and the
    landing point. This means both 'Aiming Point Range' and 'Distance To Landing'.
    The distance to the aiming point (piano keys) is used in preference but can
    fallback to using landing point distance.
    '''
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL', 'SAT', 'Approach And Landing'),
                      available) and \
               any_of(('Aiming Point Range', 'Distance To Landing'), available)

    def derive(self, alt_aal=P('Altitude AAL'),
               dist_aim=P('Aiming Point Range'),
               dist_land=P('Distance To Landing'),
               sat=P('SAT'),
               apps=S('Approach And Landing')):
        dist = dist_aim or dist_land
        self.array = np_ma_masked_zeros_like(alt_aal.array)
        for app in apps:
            if not np.ma.count(alt_aal.array[app.slice]):
                continue
            # What's the temperature deviation from ISA at landing?
            try:
                dev = from_isa(alt_aal.array[app.slice].compressed()[-1],
                               sat.array[app.slice].compressed()[-1])
            except IndexError:
                continue  # either array is entirely masked during slice

            # now correct the altitude for temperature deviation.
            alt = alt_dev2alt(alt_aal.array[app.slice], dev)

            if np.ma.any(alt>=200.0):
                alt_cropped = mask_outside_slices(alt, runs_of_ones(alt >= 200.0))
            else:
                # Altitude too low to calculate angle
                continue

            if np.min(alt_cropped) > 500:
                # Can occur in an approach to a go-around
                continue

            alt_band = runs_of_ones(alt_cropped < 500)[0]
            if alt_band.stop - alt_band.start <= 1:
                # Cannot call coreg with data of length 1
                continue

            corr, slope, offset = coreg(
                alt_aal.array[shift_slice(alt_band, app.slice.start)],
                indep_var=dist.array[shift_slice(alt_band, app.slice.start)]
            )

            dist_adj = -offset/slope
            slope_to_ldg = alt_cropped / ut.convert(
                dist.array[app.slice]-dist_adj, ut.NM, ut.FT
            )
            self.array[app.slice] = np.degrees(np.arctan(slope_to_ldg))


'''

TODO: Revise computation of sliding motion

class GroundspeedAlongTrack(DerivedParameterNode):
    """
    Inertial smoothing provides computation of groundspeed data when the
    recorded groundspeed is unreliable. For example, during sliding motion on
    a runway during deceleration. This is not good enough for long period
    computation, but is an improvement over aircraft where the groundspeed
    data stops at 40kn or thereabouts.
    """
    def derive(self, gndspd=P('Groundspeed'),
               at=P('Acceleration Along Track'),
               alt_aal=P('Altitude AAL'),
               glide = P('ILS Glideslope')):
        at_washout = first_order_washout(at.array, AT_WASHOUT_TC, gndspd.hz,
                                         gain=GROUNDSPEED_LAG_TC*GRAVITY_METRIC)
        self.array = first_order_lag(ut.convert(gndspd.array, ut.KT, ut.METER_S) + at_washout,
                                     GROUNDSPEED_LAG_TC,gndspd.hz)


        """
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        import csv
        spam = csv.writer(open('beans.csv', 'wb'))
        spam.writerow(['at', 'gndspd', 'at_washout', 'self', 'alt_aal','glide'])
        for showme in range(0, len(at.array)):
            spam.writerow([at.array.data[showme],
                           ut.convert(gndspd.array.data[showme], ut.KT, ut.FPS),
                           at_washout[showme],
                           self.array.data[showme],
                           alt_aal.array[showme],glide.array[showme]])
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        """
'''

class HeadingContinuous(DerivedParameterNode):
    '''
    For all internal computing purposes we use this parameter which does not
    jump as it passes through North. To recover the compass display, modulus
    (val % 360 in Python) returns the value to display to the user.

    Some aircraft have poor matching between captain and first officer
    signals, in which case we supply both parameters and merge here. A single
    "Heading" parameter is also required to allow initial data validation
    processes to recognise flight phases. (CRJ-100-200 is an example).
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return ('Heading' in available or
                all_of(('Heading (Capt)', 'Heading (FO)'), available))

    def derive(self, head_mag=P('Heading'),
               head_capt=P('Heading (Capt)'),
               head_fo=P('Heading (FO)'),
               frame = A('Frame')):

        frame_name = frame.value if frame else ''

        if frame_name in ['L382-Hercules']:
            gauss = [0.054488683, 0.244201343, 0.402619948, 0.244201343, 0.054488683]
            self.array = moving_average(
                straighten_headings(repair_mask(head_mag.array,
                                                repair_duration=None)),
                window=5, weightings=gauss)

        else:
            if head_capt and head_fo and (head_capt.hz==head_fo.hz):
                head_capt.array = repair_mask(straighten_headings(head_capt.array))
                head_fo.array = repair_mask(straighten_headings(head_fo.array))

                # If two compasses start up aligned east and west of North,
                # the blend_two_parameters can give a result 180 deg out. The
                # next three lines correct this error condition.
                diff = np.ma.mean(head_capt.array) - np.ma.mean(head_fo.array)
                corr = ((int(diff)+180)//360)*360.0
                head_fo.array += corr

                self.array, self.frequency, self.offset = blend_two_parameters(head_capt, head_fo)
            elif np.ma.count(head_mag.array):
                self.array = repair_mask(straighten_headings(head_mag.array))


class HeadingTrueContinuous(DerivedParameterNode):
    '''
    For all internal computing purposes we use this parameter which does not
    jump as it passes through North. To recover the compass display, modulus
    (val % 360 in Python) returns the value to display to the user.
    '''

    units = ut.DEGREE

    def derive(self, hdg=P('Heading True')):
        self.array = repair_mask(straighten_headings(hdg.array))


class Heading(DerivedParameterNode):
    '''
    Compensates for magnetic variation, which will have been computed
    previously based on the magnetic declanation at the aircraft's location.
    '''

    units = ut.DEGREE

    def derive(self, head_true=P('Heading True Continuous'),
               mag_var=P('Magnetic Variation')):
        self.array = (head_true.array - mag_var.array) % 360.0


class HeadingTrue(DerivedParameterNode):
    '''
    Compensates for magnetic variation, which will have been computed
    previously.

    The Magnetic Variation from identified Takeoff and Landing runways is
    taken in preference to that calculated based on geographical latitude and
    longitude in order to account for any compass drift or out of date
    magnetic variation databases on the aircraft.
    '''

    units = ut.DEGREE

    def derive(self, head=P('Heading Continuous'),
               rwy_var=P('Magnetic Variation From Runway')):
        var = rwy_var.array
        self.array = (head.array + var) % 360.0


class ILSFrequency(DerivedParameterNode):
    '''
    Identification of the tuned ILS Frequency.

    Where two systems are recorded, this adopts the No.1 system where
    possible, reverting to the No.2 system when this is tuned to an ILS
    frequency and No1 is not.

    Note: This code used to check for both receivers tuned to the same ILS
    frequency, but on a number of flights one receiver was found to be tuned
    to a VOR or DME, hence the change in function.
    '''

    name = 'ILS Frequency'
    align = False
    units = ut.MHZ

    @classmethod
    def can_operate(cls, available):
        return ('ILS (1) Frequency' in available and
                'ILS (2) Frequency' in available) or \
               ('ILS-VOR (1) Frequency' in available)

    def derive(self, f1=P('ILS (1) Frequency'), f2=P('ILS (2) Frequency'),
               f1v=P('ILS-VOR (1) Frequency'), f2v=P('ILS-VOR (2) Frequency')):

        #TODO: Extend to allow for three-receiver installations
        if f1 and f2:
            first = f1.array
            # align second to the first
            #TODO: Could check which is the higher frequency and align to that
            second = align(f2, f1, interpolate=False)
        elif f1v and f2v:
            first = f1v.array
            # align second to the first
            second = align(f2v, f1v, interpolate=False)
        elif f1v and not f2v:
            # Some aircraft have inoperative ILS-VOR (2) systems, which
            # record frequencies outside the valid range.
            first = f1v.array
        else:
            raise ValueError("Incorrect set of ILS frequency parameters")

        # Mask invalid frequencies
        f1_trim = filter_vor_ils_frequencies(first, 'ILS')
        if f1v and not f2v:
            self.array = f1_trim
        else:
            f2_trim = filter_vor_ils_frequencies(second, 'ILS')
            # We use getmaskarray rather than .mask to provide a correct
            # dimension array in the presence of fully valid data.
            self.array = np.ma.where(np.ma.getmaskarray(f1_trim), f2_trim, f1_trim)


class ILSLocalizer(DerivedParameterNode):
    '''
    This derived parameter merges the available sources into a single
    consolidated parameter.

    Different forms of parameter blending are used to cater for the various numbers
    of available signals on different aircraft.
    '''

    name = 'ILS Localizer'
    align = False
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               src_A=P('ILS (1) Localizer'),
               src_B=P('ILS (2) Localizer'),
               src_C=P('ILS (3) Localizer'),
               src_D=P('ILS (4) Localizer'),
               src_E=P('ILS (L) Localizer'),
               src_F=P('ILS (R) Localizer'),
               src_G=P('ILS (C) Localizer'),
               src_J=P('ILS (EFIS) Localizer'),
               ias=P('Airspeed'),
               ):

        sources = [src_A, src_B, src_C, src_D, src_E, src_F, src_G, src_J]
        active_sources = [s for s in sources if s]

        source_count = len(active_sources)
        if source_count == 0:
            # If all sources of data are masked during validation, return a null parameter
            self.offset = ias.offset
            self.frequency = ias.frequency
            self.array = np_ma_masked_zeros_like(ias.array)

        elif source_count == 1:
            self.offset = active_sources[0].offset
            self.frequency = active_sources[0].frequency
            self.array = active_sources[0].array

        elif source_count == 2:
            self.array, self.frequency, self.offset = blend_two_parameters(active_sources[0],
                                                                          active_sources[1],
                                                                          mode='localizer')

        else:
            self.offset = 0.0
            self.frequency = 2.0
            self.array = blend_parameters(sources, offset=self.offset,
                                          frequency=self.frequency)


class ILSLateralDistance(DerivedParameterNode):
    '''
    Lateral distance from the runway centreline based in ILS localizer
    signals, scaled in metres, positive to the right of the centreline.

    The term distance is used to indicate linear distance, rather than the
    angular deviation (dots) of the ILS system.
    '''

    units = ut.METER
    name = 'ILS Lateral Distance'

    def derive(self, loc=P('ILS Localizer'), app_rng=P('Approach Range'),
               approaches=App('Approach Information')):

        self.array = np_ma_masked_zeros_like(loc.array)

        for approach in approaches:
            runway = approach.approach_runway
            if not runway:
                # no runway to establish distance to localizer antenna
                continue

            try:
                start_2_loc = runway_distances(runway)[0]
                hw = ut.convert(runway['strip']['width'] / 2.0, ut.FT, ut.METER)
            except (KeyError, TypeError):
                self.warning('Unknown runway width or localizer coordinates')
                continue

            # Scale for localizer deviation to metres at runway start
            scale = hw / start_2_loc
            s = slices_int(approach.slice)
            self.array[s] = loc.array[s] * app_rng.array[s] * scale


class ILSGlideslope(DerivedParameterNode):
    '''
    This derived parameter merges the available sources into a single
    consolidated parameter. The more complex form of parameter blending is
    used to allow for many permutations.
    '''

    name = 'ILS Glideslope'
    align = False
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               src_A=P('ILS (1) Glideslope'),
               src_B=P('ILS (2) Glideslope'),
               src_C=P('ILS (3) Glideslope'),
               src_D=P('ILS (4) Glideslope'),
               src_E=P('ILS (L) Glideslope'),
               src_F=P('ILS (R) Glideslope'),
               src_G=P('ILS (C) Glideslope'),
               src_J=P('ILS (EFIS) Glideslope')):

        sources = [src_A, src_B, src_C, src_D, src_E, src_F, src_G, src_J]
        self.offset = 0.0
        self.frequency = 2.0
        self.array = blend_parameters(sources, offset=self.offset,
                                      frequency=self.frequency)


class AimingPointRange(DerivedParameterNode):
    '''
    Aiming Point Range is derived from the Approach Range. The units are
    converted to nautical miles ready for plotting and the datum is offset to
    either the ILS Glideslope Antenna position where an ILS is installed or
    the nominal threshold position where there is no ILS installation.
    '''

    units = ut.NM

    def derive(self, app_rng=P('Approach Range'),
               approaches=App('Approach Information'),
               ):
        self.array = np_ma_masked_zeros_like(app_rng.array)

        for approach in approaches:
            runway = approach.landing_runway
            if not runway:
                # no runway to establish distance to glideslope antenna
                continue
            try:
                extend = runway_distances(runway)[1] # gs_2_loc
            except (KeyError, TypeError):
                extend = runway_length(runway) - ut.convert(1000, ut.FT, ut.METER)

            s = slices_int(approach.slice)
            self.array[s] = ut.convert(app_rng.array[s] - extend, ut.METER, ut.NM)


class CoordinatesSmoothed(object):
    '''
    Superclass for SmoothedLatitude and SmoothedLongitude classes as they share
    the adjust_track methods.

    _adjust_track_pp is used for aircraft with precise positioning, usually
    GPS based and qualitatively determined by a recorded track that puts the
    aircraft on the correct runway. In these cases we only apply fine
    adjustment of the approach and landing path using ILS localizer data to
    position the aircraft with respect to the runway centreline.

    _adjust_track_ip is for aircraft with imprecise positioning. In these
    cases we use all the data available to correct for errors in the recorded
    position at takeoff, approach and landing.
    '''
    def taxi_out_track_pp(self, lat, lon, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi out track.
        '''

        lat_out, lon_out = ground_track_precise(lat, lon, speed, hdg, freq)
        return lat_out, lon_out

    def taxi_in_track_pp(self, lat, lon, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi in track.
        '''
        lat_in, lon_in = ground_track_precise(lat, lon, speed, hdg, freq)
        return lat_in, lon_in

    def taxi_out_track(self, toff_slice, lat_adj, lon_adj, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi out track.
        TODO: Include lat & lon corrections for precise positioning tracks.
        '''
        lat_out, lon_out = \
            ground_track(lat_adj[toff_slice.start],
                         lon_adj[toff_slice.start],
                         speed[:toff_slice.start],
                         hdg.array[:toff_slice.start],
                         freq,
                         'takeoff')
        return lat_out, lon_out

    def taxi_in_track(self, lat_adj, lon_adj, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi in track.
        '''
        if len(speed):
            lat_in, lon_in = ground_track(lat_adj[0],
                                          lon_adj[0],
                                          speed,
                                          hdg,
                                          freq,
                                          'landing')
            return lat_in, lon_in
        else:
            return [],[]

    def _adjust_track(self, lon, lat, ils_loc, app_range, hdg, gspd, tas,
                      toff, toff_rwy, tdwns, approaches, mobile, precise, ac_type):
        '''
        Returns track adjustment
        '''
        # Set up a working space.
        lat_adj = np_ma_masked_zeros_like(hdg.array)
        lon_adj = np_ma_masked_zeros_like(hdg.array)

        mobiles = [s.slice for s in mobile]
        begin = mobiles[0].start
        end = mobiles[-1].stop

        ils_join_offset = None

        #------------------------------------
        # Use synthesized track for takeoffs
        #------------------------------------

        # We compute the ground track using best available data.
        if gspd:
            speed = gspd.array
            freq = gspd.frequency
        else:
            speed = tas.array
            freq = tas.frequency

        try:
            toff_slice = toff[0].slice
        except:
            toff_slice = None

        if ac_type != helicopter:
            if toff_slice and precise and len(lat.array[begin:toff_slice.start]) >= 2:
                try:
                    lat_out, lon_out = self.taxi_out_track_pp(
                        lat.array[begin:toff_slice.start],
                        lon.array[begin:toff_slice.start],
                        speed[begin:toff_slice.start],
                        hdg.array[begin:toff_slice.start],
                        freq)
                except ValueError:
                    self.exception("'%s'. Using non smoothed coordinates for Taxi Out",
                                 self.__class__.__name__)
                    lat_out = lat.array[begin:toff_slice.start]
                    lon_out = lon.array[begin:toff_slice.start]
                lat_adj[begin:toff_slice.start] = lat_out
                lon_adj[begin:toff_slice.start] = lon_out

            elif toff_slice and toff_rwy and toff_rwy.value:

                toff_start_lat = lat.array[toff_slice.start]
                toff_start_lon = lon.array[toff_slice.start]
                masked_toff = any([x is np.ma.masked for x in (toff_start_lat, toff_start_lon)])
                start_locn_recorded = runway_snap_dict(
                    toff_rwy.value, toff_start_lat,
                    toff_start_lon)
                start_locn_default = toff_rwy.value['start']
                _,distance = bearing_and_distance(start_locn_recorded['latitude'],
                                                  start_locn_recorded['longitude'],
                                                  start_locn_default['latitude'],
                                                  start_locn_default['longitude'])

                if distance < 50 and not masked_toff:
                    # We may have a reasonable start location, so let's use that
                    start_locn = start_locn_recorded
                    initial_displacement = 0.0
                else:
                    # The recorded start point is way off, default to 50m down the track.
                    start_locn = start_locn_default
                    initial_displacement = 50.0

                # With imprecise navigation options it is common for the lowest
                # speeds to be masked, so we pretend to accelerate smoothly from
                # standstill.
                if speed[toff_slice][0] is np.ma.masked:
                    speed[toff_slice][0] = 0
                    speed[toff_slice] = interpolate(speed[toff_slice])

                # Compute takeoff track from start of runway using integrated
                # groundspeed, down runway centreline to end of takeoff (35ft
                # altitude). An initial value of 100m puts the aircraft at a
                # reasonable position with respect to the runway start.
                rwy_dist = np.ma.array(
                    data = integrate(speed[toff_slice], freq,
                                     initial_value=initial_displacement,
                                     extend=True,
                                     scale=ut.multiplier(ut.KT, ut.METER_S)),
                    mask = np.ma.getmaskarray(speed[toff_slice]))

                # Similarly the runway bearing is derived from the runway endpoints
                # (this gives better visualisation images than relying upon the
                # nominal runway heading). This is converted to a numpy masked array
                # of the length required to cover the takeoff phase.
                rwy_hdg = runway_heading(toff_rwy.value)
                rwy_brg = np_ma_ones_like(speed[toff_slice])*rwy_hdg

                # The track down the runway centreline is then converted to
                # latitude and longitude.
                lat_adj[toff_slice], lon_adj[toff_slice] = \
                    latitudes_and_longitudes(rwy_brg,
                                             rwy_dist,
                                             start_locn)

                lat_out, lon_out = self.taxi_out_track(toff_slice, lat_adj, lon_adj, speed, hdg, freq)

                # If we have an array holding the taxi out track, then we use
                # this, otherwise we hold at the startpoint.
                if lat_out is not None and lat_out.size:
                    lat_adj[:toff_slice.start] = lat_out
                else:
                    lat_adj[:toff_slice.start] = lat_adj[toff_slice.start]

                if lon_out is not None and lon_out.size:
                    lon_adj[:toff_slice.start] = lon_out
                else:
                    lon_adj[:toff_slice.start] = lon_adj[toff_slice.start]

            else:
                self.warning('Cannot smooth taxi out')

        #-----------------------------------------------------------------------
        # Use ILS track for approach and landings in all localizer approaches
        #-----------------------------------------------------------------------

        if not approaches:
            return lat_adj, lon_adj

        # Work through the approaches in sequence.
        for approach in approaches:

            this_app_slice = slices_int(approach.slice)

            # Set a default reference point that can be be used for Go-Arounds

            # If we really did touchdown, that is the better point to use.
            try:
                low_point = next(t.index for t in tdwns if
                                 is_index_within_slice(t.index, this_app_slice))
            except StopIteration:
                low_point_array = app_range.array[this_app_slice.start:
                                                  this_app_slice.stop - 1]
                if np.ma.count(low_point_array):
                    # Find the last valid sample
                    low_point = this_app_slice.start + np.max(np.ma.where(
                        low_point_array))
                else:
                    # No valid Approach Range samples, probably due to missing
                    # runway identification
                    # TODO: fallback low_point calculation
                    low_point = this_app_slice.stop

            if approach.type == 'LANDING':
                runway = approach.landing_runway
            elif approach.type == 'GO_AROUND' or approach.type == 'TOUCH_AND_GO':
                runway = approach.approach_runway
            else:
                raise ValueError('Unknown approach type')

            if not runway:
                continue

            # We only refine the approach track if the aircraft lands off the localizer based approach
            # and the localizer is on the runway centreline.
            if approach.loc_est and is_index_within_slice(low_point, approach.loc_est) and not approach.offset_ils:
                this_loc_slice = approach.loc_est

                # Adjust the ils data to be degrees from the reference point.
                scale = localizer_scale(runway)
                bearings = (ils_loc.array[this_loc_slice] * scale + \
                            runway_heading(runway)+180.0)%360.0

                if precise:

                    # Tweek the localizer position to be on the start:end centreline
                    localizer_on_cl = ils_localizer_align(runway)

                    # Find distances from the localizer
                    _, distances = bearings_and_distances(lat.array[this_loc_slice],
                                                          lon.array[this_loc_slice],
                                                          localizer_on_cl)


                    # At last, the conversion of ILS localizer data to latitude and longitude
                    lat_adj[this_loc_slice], lon_adj[this_loc_slice] = \
                        latitudes_and_longitudes(bearings, distances, localizer_on_cl)

                else: # Imprecise navigation but with an ILS tuned.

                    # Adjust distance units
                    distances = app_range.array[this_loc_slice]

                    # Tweek the localizer position to be on the start:end centreline
                    localizer_on_cl = ils_localizer_align(runway)

                    # At last, the conversion of ILS localizer data to latitude and longitude
                    lat_adj[this_loc_slice], lon_adj[this_loc_slice] = \
                        latitudes_and_longitudes(bearings, distances,
                                                 localizer_on_cl)

                # Alignment of the ILS Localizer Range causes corrupt first
                # samples.
                lat_adj[this_loc_slice.start] = np.ma.masked
                lon_adj[this_loc_slice.start] = np.ma.masked

                ils_join_offset = None
                if approach.type == 'LANDING' and not(approach.offset_ils or approach.runway_change):
                    # Remember where we lost the ILS, in preparation for the taxi in.
                    ils_join, _ = last_valid_sample(lat_adj[this_loc_slice])
                    if ils_join:
                        ils_join_offset = this_loc_slice.start + ils_join

            else:
                # No localizer in this approach

                if precise:
                    # Without an ILS we can do no better than copy the prepared arrray data forwards.
                    lat_adj[this_app_slice] = lat.array[this_app_slice]
                    lon_adj[this_app_slice] = lon.array[this_app_slice]
                else:
                    '''
                    We need to fix the bottom end of the descent without an
                    ILS to fix. The best we can do is put the touchdown point
                    in the right place. (An earlier version put the track
                    onto the runway centreline which looked convincing, but
                    went disasterously wrong for curving visual approaches
                    into airfields like Nice).
                    '''
                    # Adjust distance units
                    distance = np.ma.array([value_at_index(app_range.array, low_point)])
                    if not distance:
                        continue
                    bearing = np.ma.array([(runway_heading(runway)+180)%360.0])

                    # Work out the touchdown or lowest point of go-around.
                    # The reference point is the end of the runway where no ILS is available.
                    ref_point = runway['end']
                    lat_tdwn, lon_tdwn = latitudes_and_longitudes(
                        bearing, distance, ref_point)

                    lat_err = value_at_index(lat.array, low_point) - lat_tdwn
                    lon_err = value_at_index(lon.array, low_point) - lon_tdwn

                    lat_adj[this_app_slice] = lat.array[this_app_slice] - lat_err
                    lon_adj[this_app_slice] = lon.array[this_app_slice] - lon_err

            # The computation of a ground track is not ILS dependent and does
            # not depend upon knowing the runway details.
            if approach.type == 'LANDING' and ac_type == aeroplane:
                # This function returns the lowest non-None offset.
                try:
                    join_idx = int(min(filter(bool, [ils_join_offset,
                                             approach.turnoff])))
                except ValueError:
                    join_idx = None

                if join_idx and (len(lat_adj) > join_idx): # We have some room to extend over.

                    if precise:
                        # Set up the point of handover
                        lat.array[join_idx] = lat_adj[join_idx]
                        lon.array[join_idx] = lon_adj[join_idx]
                        try:
                            lat_in, lon_in = self.taxi_in_track_pp(
                                lat.array[join_idx:end],
                                lon.array[join_idx:end],
                                speed[join_idx:end],
                                hdg.array[join_idx:end],
                                freq)
                        except ValueError:
                            self.exception("'%s'. Using non smoothed coordinates for Taxi In",
                                           self.__class__.__name__)
                            lat_in = lat.array[join_idx:end]
                            lon_in = lon.array[join_idx:end]
                    else:
                        if join_idx and (len(lat_adj) > join_idx):
                            scan_back = slice(join_idx, this_app_slice.start, -1)
                            lat_join = first_valid_sample(lat_adj[scan_back])
                            lon_join = first_valid_sample(lon_adj[scan_back])
                            if lat_join.index is None or lon_join.index is None:
                                lat_in = lon_in = None
                            else:
                                join_idx -= max(lat_join.index, lon_join.index) # step back to make sure the join location is not masked.
                                lat_in, lon_in = self.taxi_in_track(
                                    lat_adj[join_idx:end],
                                    lon_adj[join_idx:end],
                                    speed[join_idx:end],
                                    hdg.array[join_idx:end],
                                    freq,
                                )

                    # If we have an array of taxi in track values, we use
                    # this, otherwise we hold at the end of the landing.
                    if lat_in is not None and np.ma.count(lat_in):
                        lat_adj[join_idx:end] = lat_in
                    else:
                        lat_adj[join_idx:end] = lat_adj[join_idx]

                    if lon_in is not None and np.ma.count(lon_in):
                        lon_adj[join_idx:end] = lon_in
                    else:
                        lon_adj[join_idx:end] = lon_adj[join_idx]

        return lat_adj, lon_adj


class LatitudeSmoothed(DerivedParameterNode, CoordinatesSmoothed):
    """
    From a prepared Latitude parameter, which may have been created by
    straightening out a recorded latitude data set, or from an estimate using
    heading and true airspeed, we now match the data to the available runway
    data. (Airspeed is included as an alternative to groundspeed so that the
    algorithm has wider applicability).

    Where possible we use ILS data to make the landing data as accurate as
    possible, and we create ground track data with groundspeed and heading if
    available.

    Once these sections have been created, the parts are 'stitched' together
    to make a complete latitude trace.

    The first parameter in the derive method is heading_continuous, which is
    always available and which should always have a sample rate of 1Hz. This
    ensures that the resulting computations yield a smoothed track with 1Hz
    spacing, even if the recorded latitude and longitude have only 0.25Hz
    sample rate.
    """

    units = ut.DEGREE

    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available, precise=A('Precise Positioning'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Longitude Prepared' in available
        required = [
            'Latitude Prepared',
            'Longitude Prepared',
            'Airspeed True',
            'Approach Information',
            'Precise Positioning',
            'FDR Takeoff Runway',
            'Mobile']
        if bool(getattr(precise, 'value', False)) is False:
            required.append('Approach Range')  # required for Imprecise non ILS approaches
        return all_of(required, available) \
               and any_of(('Heading True Continuous',
                           'Heading Continuous'), available)

    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon=P('Longitude Prepared'),
               lat=P('Latitude Prepared'),
               hdg_mag=P('Heading Continuous'),
               ils_loc=P('ILS Localizer'),
               app_range=P('Approach Range'),
               hdg_true=P('Heading True Continuous'),
               gspd_u = P('Groundspeed'),
               gspd_s = P('Groundspeed Signed'),
               tas=P('Airspeed True'),
               precise=A('Precise Positioning'),
               toff=S('Takeoff Roll Or Rejected Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'),
               tdwns = S('Touchdown'),
               approaches = App('Approach Information'),
               mobile=S('Mobile'),
               ac_type = A('Aircraft Type'),
               ):

        if ac_type == aeroplane:
            precision = bool(getattr(precise, 'value', False))
            gspd = gspd_s if gspd_s else gspd_u
            hdg = hdg_true if hdg_true else hdg_mag
            lat_adj, lon_adj = self._adjust_track(
                lon, lat, ils_loc, app_range, hdg, gspd, tas, toff, toff_rwy, tdwns,
                approaches, mobile, precision, ac_type)
            self.array = track_linking(lat.array, lat_adj)
        else:
            self.array = lat.array


class LongitudeSmoothed(DerivedParameterNode, CoordinatesSmoothed):
    """
    See Latitude Smoothed for notes.
    """

    units = ut.DEGREE
    ##align_frequency = 1.0
    ##align_offset = 0.0

    @classmethod
    def can_operate(cls, available, precise=A('Precise Positioning'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Longitude Prepared' in available
        required = [
            'Latitude Prepared',
            'Longitude Prepared',
            'Airspeed True',
            'Approach Information',
            'Precise Positioning',
            'FDR Takeoff Runway',
            'Mobile']
        if bool(getattr(precise, 'value', False)) is False:
            required.append('Approach Range')  # required for Imprecise non ILS approaches
        return all_of(required, available) \
               and any_of(('Heading True Continuous',
                           'Heading Continuous'), available)

    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon = P('Longitude Prepared'),
               lat = P('Latitude Prepared'),
               hdg_mag=P('Heading Continuous'),
               ils_loc = P('ILS Localizer'),
               app_range = P('Approach Range'),
               hdg_true = P('Heading True Continuous'),
               gspd_u = P('Groundspeed'),
               gspd_s = P('Groundspeed Signed'),
               tas = P('Airspeed True'),
               precise =A('Precise Positioning'),
               toff = S('Takeoff Roll Or Rejected Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'),
               tdwns = S('Touchdown'),
               approaches = App('Approach Information'),
               mobile=S('Mobile'),
               ac_type = A('Aircraft Type'),
               ):

        if ac_type == aeroplane:
            precision = bool(getattr(precise, 'value', False))
            gspd = gspd_s if gspd_s else gspd_u
            hdg = hdg_true if hdg_true else hdg_mag
            lat_adj, lon_adj = self._adjust_track(
                lon, lat, ils_loc, app_range, hdg, gspd, tas, toff, toff_rwy,
                tdwns, approaches, mobile, precision, ac_type)
            self.array = track_linking(lon.array, lon_adj)
        else:
            self.array = lon.array


class Mach(DerivedParameterNode):
    '''
    Mach derived from air data parameters for aircraft where no suitable Mach
    data is recorded.
    '''

    units = ut.MACH

    def derive(self, cas=P('Airspeed'), alt=P('Altitude STD Smoothed')):
        dp = cas2dp(cas.array)
        p = alt2press(alt.array)
        self.array = dp_over_p2mach(dp/p)


class MagneticVariation(DerivedParameterNode):
    '''
    This computes magnetic declination values from latitude, longitude,
    altitude and date. Uses Latitude/Longitude or
    Latitude (Coarse)/Longitude (Coarse) parameters instead of Prepared or
    Smoothed to avoid cyclical dependencies.

    Example: A Magnetic Variation of +5 deg means one adds 5 degrees to
    the Magnetic Heading to obtain the True Heading.
    '''
    # 1/4 is the minimum allowable frequency due to minimum data boundary
    # of 4 seconds.
    align_frequency = 1 / 4.0
    align_offset = 0.0
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        lat = any_of(('Latitude', 'Latitude (Coarse)'), available)
        lon = any_of(('Longitude', 'Longitude (Coarse)'), available)
        return lat and lon and all_of(('Altitude AAL', 'Start Datetime'),
                                      available)

    def derive(self, lat=P('Latitude'), lat_coarse=P('Latitude (Coarse)'),
               lon=P('Longitude'), lon_coarse=P('Longitude (Coarse)'),
               alt_aal=P('Altitude AAL'), start_datetime=A('Start Datetime')):

        lat = lat or lat_coarse
        lon = lon or lon_coarse
        mag_var_frequency = int(64 * self.frequency)
        mag_vars = []
        start_date = start_datetime.value.date() if start_datetime.value else date.today()

        # TODO: Optimize.
        for lat_val, lon_val, alt_aal_val in zip(lat.array[::mag_var_frequency],
                                                 lon.array[::mag_var_frequency],
                                                 alt_aal.array[::mag_var_frequency]):
            if np.ma.masked in (lat_val, lon_val, alt_aal_val):
                mag_vars.append(np.ma.masked)
            else:
                mag_vars.append(geomag.declination(lat_val, lon_val,
                                                   alt_aal_val,
                                                   time=start_date))

        if not any(mag_vars):
            # all masked array
            self.array = np_ma_masked_zeros_like(lat.array)
            return

        # Repair mask to avoid interpolating between masked values.
        mag_vars = repair_mask(np.ma.array(mag_vars),
                               repair_duration=None,
                               extrapolate=True)
        m = np.arange(0, len(lat.array), mag_var_frequency)
        k = min(len(m)-1,3) # ensure k is not bigger than len of m as this can occur during RTO segments
        interpolator = InterpolatedUnivariateSpline(m, mag_vars, k=k)
        interpolation_length = (len(mag_vars) - 1) * mag_var_frequency
        array = np_ma_masked_zeros_like(lat.array)
        array[:interpolation_length] = \
            interpolator(np.arange(interpolation_length))

        # Exclude masked values.
        mask = lat.array.mask | lon.array.mask | alt_aal.array.mask
        array = np.ma.masked_where(mask, array)
        # Q: Not sure of the logic behind this second mask repair.
        self.array = repair_mask(array, extrapolate=True,
                                 repair_duration=None)


class MagneticVariationFromRunway(DerivedParameterNode):
    '''
    This computes difference of local magnetic variation values on the runways
    and the Magnetic Variation parameter at the same point and
    interpolates between one airport and the next. The values at each airport
    are kept constant. Magnetic Variation is fitted this difference.

    Runways identified by approaches are not included as the aircraft may
    have drift and therefore cannot establish the heading of the runway as it
    does not land on it.

    The main idea here is that we can easily identify the ends of the runway
    and the heading of the aircraft on the runway. This allows a Heading True
    to be derived from the aircraft's perceived magnetic variation. This is
    important as some aircraft's recorded Heading (magnetic) can be based
    upon magnetic variation from out of date databases. Also, by using the
    aircraft compass values to work out the variation, we inherently
    accommodate compass drift for that day.

    Example: A Magnetic Variation of +5 deg means one adds 5 degrees to
    the Magnetic Heading to obtain the True Heading.
    '''
    units = ut.DEGREE

    def derive(self,
               mag=P('Magnetic Variation'),
               head_toff = KPV('Heading During Takeoff'),
               head_land = KPV('Heading During Landing'),
               toff_rwy = A('FDR Takeoff Runway'),
               land_rwy = A('FDR Landing Runway')):
        dev = np_ma_zeros_like(mag.array)
        dev.mask = True

        # takeoff
        tof_hdg_mag_kpv = head_toff.get_first()
        if tof_hdg_mag_kpv and toff_rwy:
            takeoff_hdg_mag = tof_hdg_mag_kpv.value
            try:
                takeoff_hdg_true = runway_heading(toff_rwy.value)
            except ValueError:
                # runway does not have coordinates to calculate true heading
                pass
            else:
                # calculate the difference magnetic variation and runway magnetic
                # variation.runway magnetic variation/declination is the difference
                # from magnetic to true heading
                dev[int(tof_hdg_mag_kpv.index)] = mag.array[int(tof_hdg_mag_kpv.index)] - \
                    heading_diff(takeoff_hdg_mag, takeoff_hdg_true)

        # landing
        ldg_hdg_mag_kpv = head_land.get_last()
        if ldg_hdg_mag_kpv and land_rwy:
            landing_hdg_mag = ldg_hdg_mag_kpv.value
            try:
                landing_hdg_true = runway_heading(land_rwy.value)
            except ValueError:
                # runway does not have coordinates to calculate true heading
                pass
            else:
                # calculate the difference magnetic variation and runway magnetic
                # variation.runway magnetic variation/declination is the difference
                # from magnetic to true heading
                dev[int(ldg_hdg_mag_kpv.index)] = mag.array[int(ldg_hdg_mag_kpv.index)] - \
                    heading_diff(landing_hdg_mag, landing_hdg_true)

        # linearly interpolate between values and extrapolate to ends of the
        # array, even if only the takeoff variation is calculated as the
        # landing variation is more likely to be the same as takeoff than 0
        # degrees (and vice versa).
        offset = interpolate(dev, extrapolate=True)
        # apply offset to Magnetic Variation
        self.array = mag.array - offset


class VerticalSpeedInertial(DerivedParameterNode):
    '''
    See 'Vertical Speed' for pressure altitude based derived parameter.

    If the aircraft records an inertial vertical speed, rename this "Vertical
    Speed Inertial - Recorded" to avoid conflict

    This routine derives the vertical speed from the vertical acceleration, the
    Pressure altitude and the Radio altitude.

    Long term errors in the accelerometers are removed by washing out the
    acceleration term with a longer time constant filter before use. The
    consequence of this is that long period movements with continued
    acceleration will be underscaled slightly. As an example the test case
    with a 1ft/sec^2 acceleration results in an increasing vertical speed of
    55 fpm/sec, not 60 as would be theoretically predicted.

    Complementary first order filters are used to combine the acceleration
    data and the height data. A high pass filter on the altitude data and a
    low pass filter on the acceleration data combine to form a consolidated
    signal.

    See also http://www.flightdatacommunity.com/inertial-smoothing.
    '''

    units = ut.FPM

    def derive(self,
               az = P('Acceleration Vertical'),
               alt_std = P('Altitude STD Smoothed'),
               alt_rad = P('Altitude Radio'),
               fast = S('Fast'),
               ac_type=A('Aircraft Type')):

        def inertial_vertical_speed(alt_std_repair, frequency, alt_rad_repair,
                                    az_repair):
            # Uses the complementary smoothing approach

            # This is the accelerometer washout term, with considerable gain.
            # The initialisation "initial_value=az_repair[0]" is very
            # important, as without this the function produces huge spikes at
            # each start of a data period.
            az_washout = first_order_washout (az_repair,
                                              AZ_WASHOUT_TC, frequency,
                                              gain=GRAVITY_IMPERIAL,
                                              initial_value=np.ma.mean(az_repair[0:40]))
            inertial_roc = first_order_lag (az_washout,
                                            VERTICAL_SPEED_LAG_TC,
                                            frequency,
                                            gain=VERTICAL_SPEED_LAG_TC)

            # We only differentiate the pressure altitude data.
            roc_alt_std = first_order_washout(alt_std_repair,
                                              VERTICAL_SPEED_LAG_TC, frequency,
                                              gain=1/VERTICAL_SPEED_LAG_TC)

            roc = (roc_alt_std + inertial_roc)
            hz = az.frequency

            # Between 100ft and the ground, replace the computed data with a
            # purely inertial computation to avoid ground effect.
            climbs = slices_from_to(alt_rad_repair, 0, 100)[1]
            # Exclude small slices (< 50ft rate of change for 2 seconds).
            # TODO: Exclude insignificant rate of change.
            climbs = slices_remove_small_slices(climbs, time_limit=2,
                                                hz=frequency)
            for n, climb in enumerate(climbs):
                # From 5 seconds before lift to 100ft
                lift_m5s = int(max(0, climb.start - 5*hz))
                up = slices_int(lift_m5s if lift_m5s >= 0 else 0, climb.stop)
                up_slope = integrate(az_washout[up], hz)
                blend_end_error = roc[climb.stop-1] - up_slope[-1]
                blend_slope = np.linspace(0.0, blend_end_error, climb.stop-climb.start)
                if ac_type != helicopter and n == 0:
                    roc[:lift_m5s] = 0.0
                roc[lift_m5s:climb.start] = up_slope[:climb.start-lift_m5s]
                roc[climb] = up_slope[climb.start-lift_m5s:] + blend_slope

                '''
                # Debug plot only.
                import matplotlib.pyplot as plt
                plt.plot(az_washout[up],'k')
                plt.plot(up_slope, 'g')
                plt.plot(roc[up],'r')
                plt.plot(alt_rad_repair[up], 'c')
                plt.show()
                plt.clf()
                plt.close()
                '''

            descents = slices_from_to(alt_rad_repair, 100, 0)[1]
            # Exclude small slices (< 50ft rate of change for 2 seconds).
            # TODO: Exclude insignificant rate of change.
            descents = slices_remove_small_slices(descents, time_limit=2,
                                                  hz=frequency)
            for n, descent in enumerate(descents):
                down = slices_int(descent.start, descent.stop+5*hz)
                down_slope = integrate(az_washout[down],
                                       hz,)
                blend = roc[down.start] - down_slope[0]
                blend_slope = np.linspace(blend, -down_slope[-1], len(down_slope))
                roc[down] = down_slope + blend_slope
                if ac_type != helicopter and n == len(descents) -1 :
                    roc[int(descent.stop+5*hz):] = 0.0

                '''
                # Debug plot only.
                import matplotlib.pyplot as plt
                plt.plot(az_washout[down],'k')
                plt.plot(down_slope,'g')
                plt.plot(roc[down],'r')
                plt.plot(blend_slope,'b')
                plt.plot(down_slope + blend_slope,'m')
                plt.plot(alt_rad_repair[down], 'c')
                plt.show()
                plt.close()
                '''

            return roc * 60.0

        # Make space for the answers
        self.array = np_ma_masked_zeros_like(alt_std.array)
        hz = az.frequency

        for speedy in fast:
            # Fix minor dropouts
            az_repair = repair_mask(az.array[speedy.slice], frequency=hz)
            alt_rad_repair = repair_mask(alt_rad.array[speedy.slice], frequency=hz,
                                             repair_duration=None,
                                             raise_entirely_masked=False)
            alt_std_repair = repair_mask(alt_std.array[speedy.slice],
                                         frequency=hz)

            # np.ma.getmaskarray ensures we have complete mask arrays even if
            # none of the samples are masked (normally returns a single
            # "False" value. We ignore the rad alt mask because we are only
            # going to use the radio altimeter values below 100ft, and short
            # transients will have been repaired. By repairing with the
            # repair_duration=None option, we ignore the masked saturated
            # values at high altitude.

            az_masked = np.ma.array(data = az_repair.data,
                                    mask = np.ma.logical_or(
                                        np.ma.getmaskarray(az_repair),
                                        np.ma.getmaskarray(alt_std_repair)))

            # We are going to compute the answers only for ranges where all
            # the required parameters are available.
            clumps = np.ma.clump_unmasked(az_masked)
            for clump in clumps:
                self.array[shift_slice(clump,speedy.slice.start)] = inertial_vertical_speed(
                    alt_std_repair[clump], az.frequency,
                    alt_rad_repair[clump], az_repair[clump])


class VerticalSpeed(DerivedParameterNode):
    '''
    The period for averaging altitude data is a trade-off between transient
    response and noise rejection.

    Some older aircraft have poor resolution, and the 4 second timebase
    leaves a noisy signal. We have inspected Hercules data, where the
    resolution is of the order of 9 ft/bit, and data from the BAe 146 where
    the resolution is 15ft and 737-6 frames with 32ft resolution. In these
    cases the wider timebase with greater smoothing is necessary, albeit at
    the expense of transient response.

    For most aircraft however, a period of 4 seconds is used. This has been
    found to give good results, and is also the value used to compute the
    recorded Vertical Speed parameter on Airbus A320 series aircraft
    (although in that case the data is delayed, and the aircraft cannot know
    the future altitudes!).
    '''

    units = ut.FPM

    @classmethod
    def can_operate(cls, available):
        return 'Altitude STD Smoothed' in available

    def derive(self, alt_std=P('Altitude STD Smoothed'), frame=A('Frame')):
        frame_name = frame.value if frame else ''

        if (frame_name == '146' or
            frame_name.startswith('747-200') or
            frame_name.startswith('737-6')):
            self.array = rate_of_change(alt_std, 11.0) * 60.0
        elif frame_name == 'L382-Hercules':
            self.array = rate_of_change(alt_std, 15.0, method='regression') * 60.0
        else:
            self.array = rate_of_change(alt_std, 4.0) * 60.0


class VerticalSpeedForFlightPhases(DerivedParameterNode):
    """
    A simple and robust vertical speed parameter suitable for identifying
    flight phases. DO NOT use this for event detection.
    """

    units = ut.FPM

    def derive(self, alt_std = P('Altitude STD Smoothed')):
        # This uses a scaled hysteresis parameter. See settings for more detail.
        threshold = HYSTERESIS_FPROC * max(1, rms_noise(alt_std.array) or 1)
        # The max(1, prevents =0 case when testing with artificial data.
        self.array = hysteresis(rate_of_change(alt_std, 6) * 60, threshold)


class Relief(DerivedParameterNode):
    """
    Also known as Terrain, this is zero at the airfields. There is a small
    cliff in mid-flight where the Altitude AAL changes from one reference to
    another, however this normally arises where Altitude Radio is out of its
    operational range, so will be masked from view.
    """

    units = ut.FT

    def derive(self, alt_aal = P('Altitude AAL'),
               alt_rad = P('Altitude Radio')):
        self.array = alt_aal.array - alt_rad.array


class CoordinatesStraighten(object):
    '''
    Superclass for LatitudePrepared and LongitudePrepared.
    '''

    def _smooth_coordinates(self, coord1, coord2, ac_type):
        """
        Acceleration along track only used to determine the sample rate and
        alignment of the resulting smoothed track parameter.

        :param coord1: Either 'Latitude' or 'Longitude' parameter.
        :type coord1: DerivedParameterNode
        :param coord2: Either 'Latitude' or 'Longitude' parameter.
        :type coord2: DerivedParameterNode
        :param ac_type: 'aeroplane' or 'helicopter'
        :type ac_type: Attribute['Aircraft Type']

        :returns: coord1 smoothed.
        :rtype: np.ma.masked_array
        """
        coord1_s = repair_mask(coord1.array, coord1.frequency, repair_duration=600)
        coord2_s = repair_mask(coord2.array, coord2.frequency, repair_duration=600)

        # Join the masks, so that we only consider positional data when both are valid:
        coord1_s.mask = np.ma.logical_or(np.ma.getmaskarray(coord1.array),
                                         np.ma.getmaskarray(coord2.array))
        coord2_s.mask = np.ma.getmaskarray(coord1_s)
        # Preload the output with masked values to keep dimension correct
        array = np_ma_masked_zeros_like(coord1_s)

        # Now we just smooth the valid sections.
        tracks = np.ma.clump_unmasked(coord1_s)
        # Skip the -180 / 180 roll over point from the smoothing
        coord1_roll_overs = 1 + np.where(
            np.ma.abs(np.ma.ediff1d(coord1_s)) > 350)[0]
        coord2_roll_overs = 1 + np.where(
            np.ma.abs(np.ma.ediff1d(coord2_s)) > 350)[0]
        # We need to apply this whether Longitude is the first or second argument,
        # as it is always used in the cost function algorithm.
        for ro in list(coord1_roll_overs)+list(coord2_roll_overs):
            tracks = slices_split(tracks, ro)
        for track in tracks:
            # Reject any data with invariant positions, i.e. sitting on stand.
            if np.ma.ptp(coord1_s[track]) > 0.0 and np.ma.ptp(coord2_s[track]) > 0.0:
                coord1_s_track, coord2_s_track, cost = \
                    smooth_track(coord1_s[track], coord2_s[track], ac_type,
                                 coord1.frequency)
                array[track] = coord1_s_track
        return array


#class LongitudePrepared(DerivedParameterNode):
    #"""
    #See Latitude Smoothed for notes.
    #"""

    #align_frequency = 1
    #units = ut.DEGREE

    #@classmethod
    #def can_operate(cls, available):
        #return any_of(('Longitude Prepared (Heading)',
                       #'Longitude Prepared (Lat Lon)'), available)

    ## Note force to 1Hz operation as latitude & longitude can be only
    ## recorded at 0.25Hz.
    #def derive(self,
               #from_latlong = P('Longitude Prepared (Lat Lon)'),
               #from_heading = P('Longitude Prepared (Heading)')):
        #self.array = from_latlong.array if from_latlong else from_heading.array


#class LongitudePreparedLatLon(DerivedParameterNode, CoordinatesStraighten):
    #"""
    #See Latitude Smoothed for notes.
    #"""
    #name = 'Longitude Prepared (Lat Lon)'
    #align_frequency = 1
    #units = ut.DEGREE

    #def derive(self,
               ## align to longitude to avoid wrap around artifacts
               #lon=P('Longitude'), lat=P('Latitude'),
               #ac_type=A('Aircraft Type')):
        #"""
        #This removes the jumps in longitude arising from the poor resolution of
        #the recorded signal.
        #"""
        #self.array = self._smooth_coordinates(lon, lat, ac_type)


class LongitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    See Latitude Smoothed for notes.
    """
    name = 'Longitude Prepared'
    align_frequency = 1
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed True',
                       'Latitude At Liftoff',
                       'Longitude At Liftoff',
                       'Latitude At Touchdown',
                       'Longitude At Touchdown'), available) and \
                any_of(('Heading', 'Heading True'), available)

    # Note force to 1Hz operation as latitude & longitude can be only
    # recorded at 0.25Hz.
    def derive(self,
               hdg_mag=P('Heading'),
               hdg_true=P('Heading True'),
               tas=P('Airspeed True'),
               gspd=P('Groundspeed'),
               alt_aal=P('Altitude AAL'),
               lat_lift=KPV('Latitude At Liftoff'),
               lon_lift=KPV('Longitude At Liftoff'),
               lat_land=KPV('Latitude At Touchdown'),
               lon_land=KPV('Longitude At Touchdown')):
        hdg = hdg_true if hdg_true else hdg_mag
        speed = gspd if gspd else tas

        _, lon_array = air_track(
            lat_lift.get_first().value, lon_lift.get_first().value,
            lat_land.get_last().value, lon_land.get_last().value,
            speed.array, hdg.array, alt_aal.array, tas.frequency)
        self.array = lon_array


#class LatitudePrepared(DerivedParameterNode):
    #"""
    #See Latitude Smoothed for notes.
    #"""

    #align_frequency = 1
    #units = ut.DEGREE

    #@classmethod
    #def can_operate(cls, available):
        #return any_of(('Latitude Prepared (Lat Lon)',
                       #'Latitude Prepared (Heading)'), available)

    ## Note force to 1Hz operation as latitude & longitude can be only
    ## recorded at 0.25Hz.
    #def derive(self,
               #from_latlong = P('Latitude Prepared (Lat Lon)'),
               #from_heading = P('Latitude Prepared (Heading)')):
        #self.array = from_latlong.array if from_latlong else from_heading.array


#class LatitudePreparedLatLon(DerivedParameterNode, CoordinatesStraighten):
    #"""
    #Creates Latitude Prepared from smoothed Latitude and Longitude parameters.
    #See Latitude Smoothed for notes.
    #"""
    #name = 'Latitude Prepared (Lat Lon)'
    #align_frequency = 1
    #units = ut.DEGREE

    ## Note force to 1Hz operation as latitude & longitude can be only
    ## recorded at 0.25Hz.
    #def derive(self,
               ## align to longitude to avoid wrap around artifacts
               #lon=P('Longitude'),
               #lat=P('Latitude'),
               #ac_type=A('Aircraft Type')):
        #self.array = self._smooth_coordinates(lat, lon, ac_type)


class LatitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    Creates 'Latitude Prepared' from Heading and Airspeed between the
    takeoff and landing locations.
    """
    name = 'Latitude Prepared'
    align_frequency = 1
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed True',
                       'Latitude At Liftoff',
                       'Longitude At Liftoff',
                       'Latitude At Touchdown',
                       'Longitude At Touchdown'), available) and \
               any_of(('Heading', 'Heading True'), available)

    # Note force to 1Hz operation as latitude & longitude can be only
    # recorded at 0.25Hz.
    def derive(self,
               hdg_mag=P('Heading'),
               hdg_true=P('Heading True'),
               tas=P('Airspeed True'),
               gspd=P('Groundspeed'),
               alt_aal=P('Altitude AAL'),
               lat_lift=KPV('Latitude At Liftoff'),
               lon_lift=KPV('Longitude At Liftoff'),
               lat_land=KPV('Latitude At Touchdown'),
               lon_land=KPV('Longitude At Touchdown')):
        hdg = hdg_true if hdg_true else hdg_mag
        speed = gspd if gspd else tas

        lat_array, _ = air_track(
            lat_lift.get_first().value, lon_lift.get_first().value,
            lat_land.get_last().value, lon_land.get_last().value,
            speed.array, hdg.array, alt_aal.array, tas.frequency)
        self.array = lat_array


class HeadingRate(DerivedParameterNode):
    '''
    Simple rate of change of heading.
    '''

    units = ut.DEGREE_S

    def derive(self, head=P('Heading Continuous')):

        # add a little hysteresis to rate of change to smooth out minor changes
        roc = rate_of_change(head, 4 if head.hz > 0.25 else 1 / head.hz * 2)
        self.array = hysteresis(roc, 0.1)
        # trouble is that we're loosing the nice 0 values, so force include!
        self.array[(self.array <= 0.05) & (self.array >= -0.05)] = 0


class Pitch(DerivedParameterNode):
    '''
    Combination of pitch signals from two sources where required.
    '''

    align = False
    units = ut.DEGREE

    def derive(self, p1=P('Pitch (1)'), p2=P('Pitch (2)')):

        self.array, self.frequency, self.offset = \
            blend_two_parameters(p1, p2)


class PitchRate(DerivedParameterNode):
    '''
    Computes rate of change of pitch attitude over a two second period.

    Comment: A two second period is used to remove excessive short period
    transients which the pilot could not realistically be asked to control.
    It also means that low sample rate data (some aircraft have
    pitch sampled at 1Hz) will still give comparable results. The drawback is
    that very brief transients, for example due to rough handling or
    turbulence, will not be detected.

    The rate_of_change algorithm was extended to allow regression
    calculation. This provides a best fit slope over the two second period,
    and so reduces the sensitivity to single samples, but tends to increase
    the peak values. As this also makes the resulting computation suffer more
    from masked values, and increases the computing load, it was decided not
    to implement this for pitch and roll rates.

    http://www.flightdatacommunity.com/calculating-pitch-rate/
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch=P('Pitch'),
               frame=A('Frame')):

        frame_name = frame.value if frame else ''

        if frame_name == 'L382-Hercules':
            self.array = rate_of_change(pitch, 8.0, method='regression')
        else:
            # See http://www.flightdatacommunity.com/blog/ for commentary on pitch rate techniques.
            self.array = rate_of_change(pitch, 2.0)


class Roll(DerivedParameterNode):
    '''
    Combination of roll signals from two sources where required.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return all_of((
            'Altitude AAL',
            'Heading Continuous',
        ), available) or \
               all_of((
                   'Roll (1)',
                   'Roll (2)',
        ), available)

    def derive(self,
               r1=P('Roll (1)'),
               r2=P('Roll (2)'),
               hdg=P('Heading Continuous'),
               alt_aal=P('Altitude AAL')):

        if r1 and r2:
            # Merge data from two sources.
            self.array, self.frequency, self.offset = \
                blend_two_parameters(r1, r2)

        else:
            # Added Beechcraft as had inoperable Roll.
            # Many Hercules aircraft do not have roll recorded. This is a
            # simple substitute, derived from examination of the roll vs
            # heading rate of aircraft with a roll sensor.
            hdg_in_air = repair_mask(
                np.ma.where(align(alt_aal, hdg)==0.0, np.ma.masked, hdg.array),
                repair_duration=None, extrapolate=True)
            self.array = 8.0 * rate_of_change_array(hdg_in_air,
                                                    hdg.hz,
                                                    width=30.0,
                                                    method='regression')
            #roll = np.ma.fix_invalid(roll, mask=False, copy=False, fill_value=0.0)
            #self.array = repair_mask(roll, repair_duration=None)
            self.frequency = hdg.frequency
            self.offset = hdg.offset

            '''
            import matplotlib.pyplot as plt
            plt.plot(align(alt_aal, hdg),'r')
            plt.plot(self.array,'b')
            plt.show()
            '''


class RollSmoothed(DerivedParameterNode):

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return 'Roll' in available

    def derive(self,
               source_A=P('Roll'),
               source_B=P('Roll (1)'),
               source_C=P('Roll (2)'),
               ):

        sources = [source_A, source_B, source_C]
        max_freq = max([s.frequency for s in sources if s])

        self.offset = 0.0
        self.frequency = max([4.0, max_freq])

        self.array = blend_parameters(sources,
                                      offset=self.offset,
                                      frequency=self.frequency,
                                      small_slice_duration=10,
                                      mode='cubic')


class PitchSmoothed(DerivedParameterNode):

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return 'Pitch' in available

    def derive(self,
               source_A=P('Pitch'),
               source_B=P('Pitch (1)'),
               source_C=P('Pitch (2)'),
               ):

        sources = [source_A, source_B, source_C]
        max_freq = max([s.frequency for s in sources if s])

        self.offset = 0.0
        self.frequency = max([4.0, max_freq])

        self.array = blend_parameters(sources,
                                      offset=self.offset,
                                      frequency=self.frequency,
                                      small_slice_duration=10,
                                      mode='cubic')


class RollRate(DerivedParameterNode):
    '''
    The computational principles here are similar to Pitch Rate; see commentary
    for that parameter.
    '''

    units = ut.DEGREE_S

    def derive(self, roll=P('Roll')):

        self.array = rate_of_change(roll, 2.0)


class RollRateForTouchdown(DerivedParameterNode):
    '''
    Unsmoothed roll rate (required for touchdown).
    '''

    units = ut.DEGREE_S

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):

        family_name = family.value if family else None

        return family_name in ('ERJ-170/175',) and (
            'Roll' in available)


    def derive(self,
               roll=P('Roll'),):
        '''
        As per Embraer's AMM - Figure 606 - Sheet 1
        Rev 52 - Nov 24/17; 200-802-A/600

        RRi = roll rate at i time instant
        R(i) = roll angle at i time instant
        R(i-1) = roll angle at i-1 time instant
        dt = delta time between i and i-1

        RRi = (R(i) - R(i-1))/dt
        '''
        roc_array = np.ma.ediff1d(roll.array)*roll.hz
        roc_array = np.insert(roc_array, 0,0,axis=0) # roll rate array too short by one - prepend a zero
        self.array = np.array(roc_array)


class RollRateAtTouchdownLimit(DerivedParameterNode):
    '''
    Maximum roll rate at touchdown for current weight.
    Applicable only for Embraer E-175.
    '''
    units = ut.DEGREE_S

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):

        family_name = family.value if family else None

        return family_name in ('ERJ-170/175',) and (
               'Gross Weight Smoothed' in available)

    def derive(self,
               gw=P('Gross Weight Smoothed'),):

        '''
        Embraer 175 - AMM 2134
        200-802-A/600
        Rev 52 - Nov 24/17

        E175 Aircraft Maintenance Manual, Roll Rate Calculation and Threshold,
        Figure 606 - Sheet 2

        The following method returns an approximation of the roll limit curve.

        For weights between 20000kg and 21999kg approximate (values returned are
        slightly below the limit) roll rate limit is:    def test_can_operate(self):
        opts = RollRateAtTouchdownLimit.get_operational_combinations()
        self.assertTrue(('Gross Weight Smoothed', 'ERJ-170/175') in opts)
        f(x) = -0.001x + 34

        For weights between 22000kg and 38000kg roll rate limit is a function:
        f(x) = -0.000375x + 20.75

        For weights between 38001kg and 40000kg we assume a limit of 6 degrees per second,
        which again, is slightly below the limit.
        '''

        self.array = np_ma_masked_zeros_like(gw.array)
        range1 = (20000 <= gw.array) & (gw.array < 22000)
        self.array[range1] = gw.array[range1] * -0.001 + 34
        range2 = (22000 <= gw.array) & (gw.array <= 38000)
        self.array[range2] = gw.array[range2] * -0.000375 + 20.75
        self.array[(38000 < gw.array) & (gw.array <= 40000)] = 6


class AccelerationNormalLimitForLandingWeight(DerivedParameterNode):
    '''
    Maximum acceleration normal at touchdown for weight.
    Applicable only for Embraer E-175.

    If landing weight is higher than 33000kg, the threshold for a
    hard landing is 1.75g

    If between 22500 and 33000, the threshold for hard landing is 2.0g

    If the landing weight is lighter than 25500 the threshold for a
    hard landing is 2.1g
    '''
    units = ut.G

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):
        family_name = family.value if family else None
        return family_name in ('ERJ-170/175',) and ('Gross Weight Smoothed' in available)

    def derive(self,
               gw=P('Gross Weight Smoothed')):

        self.array = np_ma_masked_zeros_like(gw.array)
        range1 = gw.array < 25500
        self.array[range1] = 2.1
        range2 = (25500 <= gw.array) & (gw.array <= 33300)
        self.array[range2] = 2.0
        range3 = (33300 < gw.array)
        self.array[range3] = 1.75


class AccelerationNormalLowLimitForLandingWeight(DerivedParameterNode):
    '''
    Maximum acceleration normal at touchdown for weight using the
    low load threshold.
    Applicable only for Embraer E-175.

    If landing weight is higher than 34000kg, the threshold for a
    hard landing is 1.75g

    If between 25500 and 34000, the threshold for hard landing is 2.0g

    If the landing weight is between 22500 and 25500 the threshold for
    a hard landing is a slope line from 2.2g at 22500 to 2.0g at 25500
    '''
    units = ut.G

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):
        family_name = family.value if family else None
        return family_name in ('ERJ-170/175',) and ('Gross Weight Smoothed' in available)

    def derive(self,
               gw=P('Gross Weight Smoothed')):

        self.array = np_ma_masked_zeros_like(gw.array)
        range1 = gw.array < 25500
        range_slope = (gw.array >= 22500) & (gw.array < 25500)
        self.array[range1] = 2.2
        self.array[range_slope] = np.linspace(2.2, 2.0, num=3000)[gw.array.astype(np.int)[range_slope] - 22500]
        range2 = (gw.array >= 25500) & (gw.array <= 34000)
        self.array[range2] = 2.0
        range3 = gw.array > 34000
        self.array[range3] = 1.75


class AccelerationNormalHighLimitForLandingWeight(DerivedParameterNode):
    '''
    Maximum acceleration normal at touchdown for weight using the
    high load threshold.
    Applicable only for Embraer E-175.

    If landing weight is higher than 34000kg, the threshold for a
    hard landing is 1.93g

    If between 25500 and 34000, the threshold for hard landing is 2.20g

    If the landing weight is between 22500 and 25500 the threshold for
    a hard landing is a slope line from 2.42g at 22500 to 2.2g at 25500
    '''
    units = ut.G

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):
        family_name = family.value if family else None
        return family_name in ('ERJ-170/175',) and ('Gross Weight Smoothed' in available)

    def derive(self,
               gw=P('Gross Weight Smoothed')):

        self.array = np_ma_masked_zeros_like(gw.array)
        range1 = gw.array < 25500
        range_slope = (gw.array >= 22500) & (gw.array < 25500)
        self.array[range1] = 2.42
        self.array[range_slope] = np.linspace(2.42, 2.2, num=3000)[gw.array.astype(np.int)[range_slope] - 22500]
        range2 = (gw.array >= 25500) & (gw.array <= 34000)
        self.array[range2] = 2.2
        range3 = gw.array > 34000
        self.array[range3] = 1.93


class AccelerationNormalHighLimitWithFlapsDown(DerivedParameterNode):
    '''
    Maximum acceleration normal during flight with flaps extended.
    Applicable only for B737-MAX-8.

    Normal threshold is 2.0g.

    With flaps 30 or 40:
    If gross weight is less than MLW, the threshold is 2.0g.

    If between MLW and MTOW, the threshold varies linearly from
    2.0g down to 1.5g.

    If the landing weight is higher than MTOW the threshold is 1.5g.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available,
                    family=A('Family'),):
        family_name = family.value if family else None
        return family_name in ('B737 MAX',) and \
               any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
               all_of(('Gross Weight Smoothed', 'Maximum Takeoff Weight',
                       'Maximum Landing Weight'), available)

    def derive(self,
               flap_lever=P('Flap Lever'),
               flap_synth=P('Flap Lever (Synthetic)'),
               gw=P('Gross Weight Smoothed'),
               mtow=A('Maximum Takeoff Weight'),
               mlw=A('Maximum Landing Weight')):

        flap = flap_lever or flap_synth
        # 2.0g is default value
        array = np_ma_ones_like(flap.array) * 2.0

        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        np.ma.masked_where(retracted, array, copy=False)

        # With flaps 30 or 40 and above MLW
        mtow = mtow.value
        mlw = mlw.value
        flap_30_40 = (flap.array >= 30.) & (gw.array > mlw)

        # Linearly interpolate between 2.0g at MLW and 1.5g at MTOW
        array[flap_30_40] += (gw.array[flap_30_40] - mlw) / (mtow - mlw) * (1.5 - 2.0)

        flap_30_40_above_mtow = flap_30_40 & (gw.array > mtow)
        array[flap_30_40_above_mtow] = 1.5

        self.array = array


class Rudder(DerivedParameterNode):
    '''
    Combination of multi-part rudder elements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Rudder (Upper)',
            'Rudder (Middle)',
            'Rudder (Lower)',
        ), available)

    def derive(self,
               src_A=P('Rudder (Upper)'),
               src_B=P('Rudder (Middle)'),
               src_C=P('Rudder (Lower)'),
               ):

        sources = [s for s in (src_A, src_B, src_C) if s is not None]
        self.offset = 0.0
        self.frequency = sources[0].frequency
        self.array = blend_parameters(sources, offset=self.offset,
                                      frequency=self.frequency)


class RudderPedalCapt(DerivedParameterNode):
    '''
    '''

    name = 'Rudder Pedal (Capt)'
    units = ut.DEGREE  # FIXME: Or should this be ut.PERCENT?


    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Rudder Pedal (Capt) (1)',
            'Rudder Pedal (Capt) (2)',
        ), available)

    def derive(self, rudder_pedal_capt_1=P('Rudder Pedal (Capt) (1)'),
               rudder_pedal_capt_2=P('Rudder Pedal (Capt) (2)')):

        self.array, self.frequency, self.offset = \
            blend_two_parameters(rudder_pedal_capt_1, rudder_pedal_capt_2)


class RudderPedalFO(DerivedParameterNode):
    '''
    '''

    name = 'Rudder Pedal (FO)'
    units = ut.DEGREE  # FIXME: Or should this be ut.PERCENT?

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Rudder Pedal (FO) (1)',
            'Rudder Pedal (FO) (2)',
        ), available)

    def derive(self, rudder_pedal_fo_1=P('Rudder Pedal (FO) (1)'),
               rudder_pedal_fo_2=P('Rudder Pedal (FO) (2)')):

        self.array, self.frequency, self.offset = \
            blend_two_parameters(rudder_pedal_fo_1, rudder_pedal_fo_2)


class RudderPedal(DerivedParameterNode):
    '''
    '''

    units = ut.DEGREE  # FIXME: Or should this be ut.PERCENT?

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Rudder Pedal (Capt)',
            'Rudder Pedal (FO)',
            'Rudder Pedal Potentiometer',
            'Rudder Pedal Synchro',
        ), available)

    def derive(self, rudder_pedal_capt=P('Rudder Pedal (Capt)'),
               rudder_pedal_fo=P('Rudder Pedal (FO)'),
               pot=P('Rudder Pedal Potentiometer'),
               synchro=P('Rudder Pedal Synchro')):

        if rudder_pedal_capt or rudder_pedal_fo:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(rudder_pedal_capt, rudder_pedal_fo)

        synchro_samples = 0

        if synchro:
            synchro_samples = np.ma.count(synchro.array)
            self.frequency = synchro.frequency
            self.offset = synchro.offset
            self.array = synchro.array

        if pot:
            pot_samples = np.ma.count(pot.array)
            if pot_samples > synchro_samples:
                self.frequency = pot.frequency
                self.offset = pot.offset
                self.array = pot.array


class RudderPedalForce(DerivedParameterNode):
    '''
    Introduced for the CRJ fleet, where left and right pedal forces for each
    pilot are measured. We allow for both pilots pushing on the pedals at the
    same time, and make the positive = heading right sign convention to merge
    both. If you just rest your feet on the footrests, the resultant should
    be zero.
    '''

    units = ut.DECANEWTON

    def derive(self,
               fcl=P('Rudder Pedal Force (Capt) (L)'),
               fcr=P('Rudder Pedal Force (Capt) (R)'),
               ffl=P('Rudder Pedal Force (FO) (L)'),
               ffr=P('Rudder Pedal Force (FO) (R)')):

        right = fcr.array + ffr.array
        left = fcl.array + ffl.array
        self.array = right - left


class ThrottleLevers(DerivedParameterNode):
    '''
    A synthetic throttle lever angle, based on the average of the two. Allows
    for simple identification of changes in power etc.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Eng (1) Throttle Lever',
            'Eng (2) Throttle Lever',
        ), available)

    def derive(self,
               tla1=P('Eng (1) Throttle Lever'),
               tla2=P('Eng (2) Throttle Lever')):

        self.array, self.frequency, self.offset = \
            blend_two_parameters(tla1, tla2)


class ThrustAsymmetry(DerivedParameterNode):
    '''
    Thrust asymmetry based on N1.

    For EPR rated aircraft, this measure is applicable as we are
    not applying a manufacturer's limit to the value, rather this is being
    used to identify imbalance of thrust and as the thrust comes from engine
    speed, N1 is still applicable.

    Using a 5 second moving average to desensitise the parameter against
    transient differences as engines accelerate.

    If we have EPR rated engines, we treat EPR=2.0 as 100% and EPR=1.0 as 0%
    so the Thrust Asymmetry is simply (EPRmax-EPRmin)*100.

    For propeller aircraft the product of prop speed and torgue should be
    used to provide a similar single asymmetry value.
    '''

    align_frequency = 1 # Forced alignment to allow fixed window period.
    align_offset = 0
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Eng (*) EPR Max', 'Eng (*) EPR Min'), available) or\
               all_of(('Eng (*) N1 Max', 'Eng (*) N1 Min'), available)

    def derive(self, epr_max=P('Eng (*) EPR Max'), epr_min=P('Eng (*) EPR Min'),
               n1_max=P('Eng (*) N1 Max'), n1_min=P('Eng (*) N1 Min')):
        # use N1 if we have it in preference to EPR
        if n1_max:
            diff = (n1_max.array - n1_min.array)
        else:
            diff = (epr_max.array - epr_min.array) * 100
        window = 5 # 5 second window
        self.array = moving_average(diff, window=window)


class Turbulence(DerivedParameterNode):
    '''
    Turbulence is measured as the Root Mean Squared (RMS) of the Vertical
    Acceleration over a 5-second period.
    '''

    units = ut.G

    def derive(self, acc=P('Acceleration Vertical')):
        width = int(acc.frequency*5)
        width += 1 - width % 2
        mean = moving_average(acc.array, window=width)
        acc_sq = (acc.array)**2.0
        n__sum_sq = moving_average(acc_sq, window=width)
        # Rescaling required as moving average is over width samples, whereas
        # we have only width - 1 gaps; fences and fence posts again !
        core = (n__sum_sq - mean**2.0)*width/(width-1.0)
        self.array = np.ma.sqrt(core)


#------------------------------------------------------------------
# WIND RELATED PARAMETERS
#------------------------------------------------------------------


class WindDirectionContinuous(DerivedParameterNode):
    '''
    Like the aircraft heading, this does not jump as it passes through North.
    '''

    units = ut.DEGREE

    def derive(self, wind_head=P('Wind Direction'),):

        self.array = straighten_headings(wind_head.array)


class WindDirectionTrueContinuous(DerivedParameterNode):
    '''
    Like the aircraft heading, this does not jump as it passes through North.

    This is a copy of the above parameter - Wind Direction Continuous.
    We need to keep that for now as some data exports use Wind Direction True
    Continuous.

    Previously this parameter was based on Wind Direction + magnetic variation,
    and was created in assumption that Wind Direction was magnetic, not true,
    which has been proven to be incorrect.
    '''

    units = ut.DEGREE

    def derive(self, wind_dir_cont=P('Wind Direction Continuous'),):

        self.array = wind_dir_cont.array


class WindDirectionMagneticContinuous(DerivedParameterNode):
    '''
    Like the aircraft heading, this does not jump as it passes through North.
    '''

    units = ut.DEGREE

    def derive(self, wind_head=P('Wind Direction Magnetic')):

        self.array = straighten_headings(wind_head.array)


class Headwind(DerivedParameterNode):
    '''
    Headwind calculates the headwind component based upon the Wind Speed and
    Wind Direction compared to the Heading to get the direct Headwind
    component.

    If Airspeed True and Groundspeed are available, below 100ft AAL the
    difference between the two is used, ignoring the Wind Speed / Direction
    component which become erroneous.

    Negative values of this Headwind component are a Tailwind.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        wind_based = all_of((
            'Wind Speed',
            'Wind Direction',
            'Heading True',
            'Altitude AAL',
        ), available)

        aspd_based = all_of((
            'Airspeed True',
            'Groundspeed'
        ), available)

        return wind_based or aspd_based

    def derive(self, aspd=P('Airspeed True'),
               windspeed=P('Wind Speed'),
               wind_dir=P('Wind Direction'),
               head=P('Heading True'),
               alt_aal=P('Altitude AAL'),
               gspd=P('Groundspeed')):

        if all((windspeed, wind_dir, head, alt_aal)):
            # Wind based
            if aspd:
                # mask windspeed data while going slow
                windspeed.array[aspd.array.mask] = np.ma.masked
            rad_scale = radians(1.0)
            headwind = windspeed.array * np.ma.cos((wind_dir.array-head.array)*rad_scale)

            # If we have airspeed and groundspeed, overwrite the values for the
            # altitudes below one hundred feet.
            if aspd and gspd:
                for below_100ft in alt_aal.slices_below(100):
                    try:
                        headwind[below_100ft] = moving_average((aspd.array[below_100ft] - gspd.array[below_100ft]),
                                                               window=5)
                    except:
                        pass # Leave the data unchanged as one of the parameters is fully masked.
            self.array = headwind

        else:
            # Airspeed based
            try:
                headwind = moving_average((aspd.array - gspd.array),
                                          window=5)
            except ValueError:
                # one of the parameters is fully masked.
                headwind = np_ma_masked_zeros_like(aspd.array)

            self.array = headwind


class Tailwind(DerivedParameterNode):
    '''
    This is the tailwind component.
    '''

    units = ut.KT

    def derive(self, hwd=P('Headwind')):

        self.array = -hwd.array


class SAT(DerivedParameterNode):
    '''
    Computes Static Air Temperature (temperature of the outside air) from the
    Total Air Temperature, allowing for compressibility effects, or if this
    is not available, the standard atmosphere and lapse rate.
    '''

    # Q: Support transforming SAT from OAT (as they are equal).
    # TODO: Review naming convention - rename to "Static Air Temperature"?

    name = 'SAT'
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available):
        return any_of (('SAT (1)', 'SAT (2)', 'SAT (3)'), available) or all_of(('TAT', 'Mach'), available)

    def derive(self,
               sat1=P('SAT (1)'),
               sat2=P('SAT (2)'),
               sat3=P('SAT (3)'),
               tat=P('TAT'),
               mach=P('Mach')):
        sat_params = [p for p in (sat1, sat2, sat3) if p is not None]
        if sat_params:
            sats = vstack_params(*sat_params)
            self.array = np.ma.average(sats, axis=0)
            # Use average offset of the sat parameters
            self.offset = sum(p.offset for p in sat_params) / len(sat_params)
        else:
            self.array = machtat2sat(mach.array, tat.array)


class SAT_ISA(DerivedParameterNode):
    '''
    Computes Static Air Temperature (temperature of the outside air) from the
    standard atmosphere and lapse rate.
    '''

    name = 'SAT International Standard Atmosphere'
    units = ut.CELSIUS

    def derive(self, alt=P('Altitude STD Smoothed')):
        self.array = alt2sat(alt.array)


class TAT(DerivedParameterNode):
    '''
    Operates in two different modes, depending upon available information.

    (1) Blends data from two air temperature sources.
    (2) Computes TAT from SAT and Mach number.
    '''

    @classmethod
    def can_operate(cls, available):
        return (('TAT (1)' in available and 'TAT (2)' in available) or \
                ('SAT' in available and 'Mach' in available))

    # TODO: Review naming convention - rename to "Total Air Temperature"?

    name = 'TAT'
    align = False
    units = ut.CELSIUS

    def derive(self,
               source_1=P('TAT (1)'),
               source_2=P('TAT (2)'),
               sat=P('SAT'), mach=P('Mach')):

        if sat:
            # We compute the total air temperature, assuming a perfect sensor.
            # Where Mach is masked we use SAT directly
            if sat.hz > mach.hz:
                self.hz = sat.hz
                self.offset = sat.offset
                mach = mach.get_aligned(sat)
            elif mach.hz > sat.hz:
                self.hz = mach.hz
                self.offset = mach.offset
                sat = sat.get_aligned(mach)

            self.array = np.ma.where(
                np.ma.getmaskarray(mach.array), sat.array, machsat2tat(mach.array, sat.array,
                                                                       recovery_factor=1.0))

        else:
            # Alternate samples (1)&(2) are blended.
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_1, source_2)


class WindAcrossLandingRunway(DerivedParameterNode):
    '''
    This is the windspeed across the final landing runway, positive wind from
    left to right.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway'), available) \
               or \
               all_of(('Wind Speed', 'Wind Direction Magnetic Continuous', 'Heading During Landing'), available)

    def derive(self, windspeed=P('Wind Speed'),
               wind_dir_true=P('Wind Direction Continuous'),
               wind_dir_mag=P('Wind Direction Magnetic Continuous'),
               land_rwy=A('FDR Landing Runway'),
               land_hdg=KPV('Heading During Landing')):

        if wind_dir_true and land_rwy:
            # proceed with "True" values
            wind_dir = wind_dir_true
            land_heading = runway_heading(land_rwy.value)
            self.array = np_ma_masked_zeros_like(wind_dir_true.array)
        elif wind_dir_mag and land_hdg:
            # proceed with "Magnetic" values
            wind_dir = wind_dir_mag
            land_heading = land_hdg.get_last().value
        else:
            # either no landing runway detected or no landing heading detected
            self.array = np_ma_masked_zeros_like(windspeed.array)
            self.warning('Cannot calculate without landing runway (%s) or landing heading (%s)',
                         bool(land_rwy), bool(land_hdg))
            return
        diff = (land_heading - wind_dir.array) * deg2rad
        self.array = windspeed.array * np.ma.sin(diff)


class Aileron(DerivedParameterNode):
    '''
    Aileron measures the roll control from the Left and Right Aileron
    signals. By taking the average of the two signals, any Flaperon movement
    is removed from the signal, leaving only the difference between the left
    and right which results in the roll control.

    Note: This requires that both Aileron signals have positive sign for
    positive (right) rolling moment. That is, port aileron down and starboard
    aileron up have positive sign.

    Note: This is NOT a multistate parameter - see Flaperon.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Aileron (L)', 'Aileron (R)'), available)

    def derive(self, al=P('Aileron (L)'), ar=P('Aileron (R)')):
        if al and ar:
            # Taking the average will ensure that positive roll to the right
            # on both signals is maintained as positive control, where as
            # any flaperon action (left positive, right negative) will
            # average out as no roll control.
            faster, slower = sorted((al, ar), key=attrgetter('frequency'),
                                    reverse=True)
            self.frequency = faster.frequency * 2
            self.offset = (faster.offset + slower.offset) / 4
            self.array = (align(faster, self) + align(slower, self)) / 2
        else:
            ail = al or ar
            self.array = ail.array
            self.frequency = ail.frequency
            self.offset = ail.offset


class AileronLeft(DerivedParameterNode):
    '''
    '''

    name = 'Aileron (L)'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Aileron (L) Potentiometer',
                       'Aileron (L) Synchro',
                       'Aileron (L) Inboard',
                       'Aileron (L) Outboard'), available)

    def derive(self, pot=P('Aileron (L) Potentiometer'),
               synchro=P('Aileron (L) Synchro'),
               ali=P('Aileron (L) Inboard'),
               alo=P('Aileron (L) Outboard')):
        synchro_noise = 0
        if synchro:
            synchro_noise = rms_noise(synchro.array)
            self.array = synchro.array
        if pot:
            pot_noise = rms_noise(pot.array)
            if pot_noise>synchro_noise:
                self.array = pot.array
        # If Inboard available, use this in preference
        if ali:
            self.array = ali.array
        elif alo:
            self.array = alo.array


class AileronRight(DerivedParameterNode):
    '''
    '''

    name = 'Aileron (R)'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Aileron (R) Potentiometer',
                       'Aileron (R) Synchro',
                       'Aileron (R) Inboard',
                       'Aileron (R) Outboard'), available)

    def derive(self, pot=P('Aileron (R) Potentiometer'),
               synchro=P('Aileron (R) Synchro'),
               ari=P('Aileron (R) Inboard'),
               aro=P('Aileron (R) Outboard')):

        synchro_noise = 0
        if synchro:
            synchro_noise = rms_noise(synchro.array)
            self.array = synchro.array
        if pot:
            pot_noise = rms_noise(pot.array)
            if pot_noise>synchro_noise:
                self.array = pot.array
        # If Inboard available, use this in preference
        if ari:
            self.array = ari.array
        elif aro:
            self.array = aro.array


class AileronTrim(DerivedParameterNode):  # FIXME: RollTrim
    '''
    '''

    # TODO: TEST

    name = 'Aileron Trim'  # FIXME: Roll Trim
    align = False
    units = ut.DEGREE

    def derive(self,
               atl=P('Aileron Trim (L)'),
               atr=P('Aileron Trim (R)')):

        self.array, self.frequency, self.offset = blend_two_parameters(atl, atr)


class Elevator(DerivedParameterNode):
    '''
    Blends alternate elevator samples. If either elevator signal is invalid,
    this reverts to just the working sensor.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls,available):
        return any_of(('Elevator (L)', 'Elevator (R)'), available)

    def derive(self,
               el=P('Elevator (L)'),
               er=P('Elevator (R)')):

        if el and er:
            self.array, self.frequency, self.offset = blend_two_parameters(el, er)
        else:
            self.array = el.array if el else er.array
            self.frequency = el.frequency if el else er.frequency
            self.offset = el.offset if el else er.offset


class ElevatorLeft(DerivedParameterNode):
    '''
    Specific to a group of ATR aircraft which were progressively modified to
    replace potentiometers with synchros. The data validity tests will mark
    whole parameters invalid, or if both are valid, we want to pick the best
    option.

    Extended to cater for aircraft with split elevators instrumented separately.
    '''

    name = 'Elevator (L)'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Elevator (L) Potentiometer',
                       'Elevator (L) Synchro',
                       'Elevator (L) Inboard',
                       'Elevator (L) Outboard'), available)

    def derive(self, pot=P('Elevator (L) Potentiometer'),
               synchro=P('Elevator (L) Synchro'),
               inboard=P('Elevator (L) Inboard'),
               outboard=P('Elevator (L) Outboard')):

        synchro_samples = 0

        if synchro:
            synchro_samples = np.ma.count(synchro.array)
            self.array = synchro.array

        if pot:
            pot_samples = np.ma.count(pot.array)
            if pot_samples>synchro_samples:
                self.array = pot.array

        # If Inboard available, use this in preference
        if inboard:
            self.array = inboard.array
        elif outboard:
            self.array = outboard.array


class ElevatorRight(DerivedParameterNode):
    '''
    '''

    name = 'Elevator (R)'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Elevator (R) Potentiometer',
                       'Elevator (R) Synchro',
                       'Elevator (R) Inboard',
                       'Elevator (R) Outboard'), available)

    def derive(self, pot=P('Elevator (R) Potentiometer'),
               synchro=P('Elevator (R) Synchro'),
               inboard=P('Elevator (R) Inboard'),
               outboard=P('Elevator (R) Outboard')):

        synchro_samples = 0
        if synchro:
            synchro_samples = np.ma.count(synchro.array)
            self.array = synchro.array
        if pot:
            pot_samples = np.ma.count(pot.array)
            if pot_samples>synchro_samples:
                self.array = pot.array

        # If Inboard available, use this in preference
        if inboard:
            self.array = inboard.array
        elif outboard:
            self.array = outboard.array

##############################################################################
# Speedbrake


class Speedbrake(DerivedParameterNode):
    '''
    Spoiler angle in degrees, zero flush with the wing and positive up.

    Spoiler positions are recorded in different ways on different aircraft,
    hence the frame specific sections in this class.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        '''
        Note: The frame name cannot be accessed within this method to determine
              which parameters are required.

        For B737-NG aircraft, D226A101-2 RevH states:
            NOTE 25D Spoiler Position No. 3 and 10
            (737-3, -3A, -3B, -3C, -7)
            For airplanes that do not have the Short Field Performance option
        This suggests that the synchro sourced L&R 3 positions have a scaling
        that changes with short field option.

        CL-600 has 3 operational combinations;
         * CRJ100/200/Challenger 850 Flight Spoiler is L&R 3
         * CRJ700/900 Flight Spoilers are L&R 3 and 4
         * Challenger 605 Flight Spoiler is L&R 2
        '''
        family_name = family.value if family else None
        return family_name and (
            family_name in ('G-V', 'G-IV', 'CL-600') and (
                'Spoiler (L) (2)' in available and
                'Spoiler (R) (2)' in available
            ) or
            family_name in ('G-VI',) and (
                'Spoiler (L) (1)' in available and
                'Spoiler (R) (1)' in available
            ) or
            family_name in ('A300', 'A318', 'A319', 'A320', 'A321', 'A330', 'A340', 'A380') and (
                ('Spoiler (L) (3)' in available and
                    'Spoiler (R) (3)' in available) or
                ('Spoiler (L) (2)' in available and
                    'Spoiler (R) (2)' in available)
            ) or
            family_name in ( 'A350', ) and (
                'Spoiler (L) (4)' in available and
                'Spoiler (R) (4)' in available
            ) or
            family_name in ('B737 Classic',) and (
                'Spoiler (L) (4)' in available and
                'Spoiler (R) (4)' in available
            ) or
            family_name in ('B737 NG', 'B737 MAX') and (
                ('Spoiler (L) (3)' in available and
                    'Spoiler (R) (3)' in available) or
                ('Spoiler (L) (4)' in available and
                    'Spoiler (R) (4)' in available)
            ) or
            family_name == 'Global' and (
                'Spoiler (L) (5)' in available and
                'Spoiler (R) (5)' in available
            ) or
            family_name == 'B787' and (
                'Spoiler (L) (7)' in available and
                'Spoiler (R) (7)' in available
            ) or
            family_name in ('Learjet', 'Phenom 300', 'Citation', 'Citation VLJ') and all_of((
                'Spoiler (L)',
                'Spoiler (R)'),
                available
            ) or
            family_name in ['CRJ 900', 'CL-600', 'G-IV'] and all_of((
                'Spoiler (L) (3)',
                'Spoiler (L) (4)',
                'Spoiler (R) (3)',
                'Spoiler (R) (4)'),
                available
            ) or
            family_name == 'CL-600' and all_of((
                'Spoiler (L) (3)',
                'Spoiler (R) (3)',),
                available
            ) or
            family_name == 'MD-11' and all_of((
                'Spoiler (L) (3)',
                'Spoiler (L) (5)',
                'Spoiler (R) (3)',
                'Spoiler (R) (5)'),
                available
            ) or
            family_name in ['ERJ-170/175', 'ERJ-190/195'] and all_of((
                'Spoiler (L) (3)',
                'Spoiler (L) (4)',
                'Spoiler (L) (5)',
                'Spoiler (R) (3)',
                'Spoiler (R) (4)',
                'Spoiler (R) (5)'),
                available
            ) or
            family_name in ['B777'] and all_of((
                'Spoiler (L) (6)',
                'Spoiler (L) (7)',
                'Spoiler (R) (6)',
                'Spoiler (R) (7)'),
                available
            )
        )

    def merge_spoiler(self, spoiler_a, spoiler_b):
        '''
        We indicate the angle of the lower of the two raised spoilers, as
        this represents the drag element. Differential deployment is used to
        augments roll control, so the higher of the two spoilers is relating
        to roll control. Small values are ignored as these arise from control
        trim settings.
        '''
        assert spoiler_a.frequency == spoiler_b.frequency, \
               "Cannot merge Spoilers of differing frequencies"
        self.frequency = spoiler_a.frequency
        self.offset = (spoiler_a.offset + spoiler_b.offset) / 2.0
        array = np.ma.minimum(spoiler_a.array, spoiler_b.array)
        # Force small angles to indicate zero:
        self.array = np.ma.where(array < 10.0, 0.0, array)

    def derive(self,
               spoiler_l=P('Spoiler (L)'),
               spoiler_l1=P('Spoiler (L) (1)'),
               spoiler_l2=P('Spoiler (L) (2)'),
               spoiler_l3=P('Spoiler (L) (3)'),
               spoiler_l4=P('Spoiler (L) (4)'),
               spoiler_l5=P('Spoiler (L) (5)'),
               spoiler_l6=P('Spoiler (L) (6)'),
               spoiler_l7=P('Spoiler (L) (7)'),
               spoiler_r=P('Spoiler (R)'),
               spoiler_r1=P('Spoiler (R) (1)'),
               spoiler_r2=P('Spoiler (R) (2)'),
               spoiler_r3=P('Spoiler (R) (3)'),
               spoiler_r4=P('Spoiler (R) (4)'),
               spoiler_r5=P('Spoiler (R) (5)'),
               spoiler_r6=P('Spoiler (R) (6)'),
               spoiler_r7=P('Spoiler (R) (7)'),
               family=A('Family'),
               ):

        family_name = family.value

        if family_name in ('G-V', 'G-IV') or (family_name == 'CL-600' and spoiler_l2 and spoiler_r2):
            self.merge_spoiler(spoiler_l2, spoiler_r2)
        elif family_name in ('G-VI',):
            self.merge_spoiler(spoiler_l1, spoiler_r1)
        elif family_name in ('A300', 'A318', 'A319', 'A320', 'A321', 'A330', 'A340', 'A380'):
            if spoiler_l3 is not None:
                self.merge_spoiler(spoiler_l3, spoiler_r3)
            else:
                self.merge_spoiler(spoiler_l2, spoiler_r2)
        elif family_name in ('A350',):
            self.merge_spoiler(spoiler_l4, spoiler_r4)
        elif family_name in ('B737 Classic',):
            self.merge_spoiler(spoiler_l4, spoiler_r4)
        elif family_name in ('B737 NG', 'B737 MAX'):
            if spoiler_l3 and spoiler_r3:
                self.merge_spoiler(spoiler_l3, spoiler_r3)
            else:
                self.merge_spoiler(spoiler_l4, spoiler_r4)
        elif family_name == 'Global':
            self.merge_spoiler(spoiler_l5, spoiler_r5)
        elif family_name == 'B787':
            self.merge_spoiler(spoiler_l7, spoiler_r7)
        elif family_name in ('Learjet', 'Phenom 300', 'Citation', 'Citation VLJ'):
            self.merge_spoiler(spoiler_l, spoiler_r)
        elif family_name in ('CRJ 900', 'CL-600', 'G-IV'):
            # First blend inboard and outboard, then merge
            spoiler_L = DerivedParameterNode(
                'Spoiler (L)', *blend_two_parameters(spoiler_l3, spoiler_l4))
            spoiler_R = DerivedParameterNode(
                'Spoiler (R)', *blend_two_parameters(spoiler_r3, spoiler_r4))
            self.merge_spoiler(spoiler_L, spoiler_R)
        elif family_name == 'MD-11':
            # First blend inboard and outboard, then merge
            spoiler_L = DerivedParameterNode(
                'Spoiler (L)', *blend_two_parameters(spoiler_l3, spoiler_l5))
            spoiler_R = DerivedParameterNode(
                'Spoiler (R)', *blend_two_parameters(spoiler_r3, spoiler_r5))
            self.merge_spoiler(spoiler_L, spoiler_R)
        elif family_name in ('ERJ-170/175', 'ERJ-190/195'):
            # First blend inboard, middle and outboard, then merge
            spoiler_L = DerivedParameterNode(
                'Spoiler (L)',
                blend_parameters((spoiler_l3, spoiler_l4, spoiler_l5)))
            spoiler_R = DerivedParameterNode(
                'Spoiler (R)',
                blend_parameters((spoiler_r3, spoiler_r4, spoiler_r5)))
            self.merge_spoiler(spoiler_L, spoiler_R)
        elif family_name in ('B777',):
            spoiler_L = DerivedParameterNode(
                'Spoiler (L)',
                *blend_two_parameters(spoiler_l6, spoiler_l7))
            spoiler_R = DerivedParameterNode(
                'Spoiler (R)',
                *blend_two_parameters(spoiler_r6, spoiler_r7))
            self.merge_spoiler(spoiler_L, spoiler_R)
        else:
            raise DataFrameError(self.name, family_name)


class SpeedbrakeHandle(DerivedParameterNode):
    '''
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Speedbrake Handle (L)',
            'Speedbrake Handle (R)',
            'Speedbrake Handle (C)',
            'Speedbrake Handle (1)',
            'Speedbrake Handle (2)',
            'Speedbrake Handle (3)',
            'Speedbrake Handle (4)',
        ), available)

    def derive(self,
               sbh_l=P('Speedbrake Handle (L)'),
               sbh_r=P('Speedbrake Handle (R)'),
               sbh_c=P('Speedbrake Handle (C)'),
               sbh_1=P('Speedbrake Handle (1)'),
               sbh_2=P('Speedbrake Handle (2)'),
               sbh_3=P('Speedbrake Handle (3)'),
               sbh_4=P('Speedbrake Handle (4)')):

        available = [par for par in [sbh_l, sbh_r, sbh_c, sbh_1, sbh_2, sbh_3, sbh_4] if par]
        if len(available) > 1:
            self.array = blend_parameters(
                available, self.offset, self.frequency)
        elif len(available) == 1:
            self.array = available[0].array


class Stabilizer(DerivedParameterNode):
    '''
    Combination of multi-part stabilizer elements.

    Three sensors measure the input shaft angle, converted here for the 777 surface.

    See D247W018-9 Page 2677
    '''

    units = ut.DEGREE

    def derive(self,
               src_1=P('Stabilizer (1)'),
               src_2=P('Stabilizer (2)'),
               src_3=P('Stabilizer (3)'),
               frame = A('Frame'),
               ):

        frame_name = frame.value if frame else ''

        if frame_name == '777':
            sources = [src_1, src_2, src_3]
            self.offset = 0.0
            self.frequency = src_1.frequency
            shaft_angle = blend_parameters(sources, offset=self.offset,
                                           frequency=self.frequency)
            self.array = 0.0503 * shaft_angle - 3.4629
        else:
            raise ValueError('Stabilizer called but not for 777 frame.')


class ApproachRange(DerivedParameterNode):
    '''
    This is the range to the touchdown point for both ILS and visual
    approaches including go-arounds. The reference point is the ILS Localizer
    antenna where the runway is so equipped, or the end of the runway where
    no ILS is available.

    The array is masked where no data has been computed, and provides
    measurements in metres from the reference point where the aircraft is on
    an approach.

    A simpler function is provided for helicopter operations as they may
    not - in fact normally do not - have a runway to land on.
    '''

    units = ut.METER

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed True',
                       'Altitude AAL',
                       'Approach Information'), available) and \
               any_of(('Heading True',
                       'Track True',
                       'Track',
                       'Heading'), available)

    def derive(self, gspd=P('Groundspeed'),
               glide=P('ILS Glideslope'),
               trk_mag=P('Track'),
               trk_true=P('Track True'),
               hdg_mag=P('Heading'),
               hdg_true=P('Heading True'),
               tas=P('Airspeed True'),
               alt_aal=P('Altitude AAL'),
               approaches=App('Approach Information'),
               ):
        app_range = np_ma_masked_zeros_like(alt_aal.array)

        for approach in approaches:
            # We are going to reference the approach to a runway touchdown
            # point. Without that it's pretty meaningless, so give up now.
            runway = approach.landing_runway
            if not runway:
                continue

            # Retrieve the approach slice
            this_app_slice = slices_int(approach.slice)

            # Let's use the best available information for this approach
            if trk_true and np.ma.count(trk_true.array[this_app_slice]):
                hdg = trk_true
                magnetic = False
            elif trk_mag and np.ma.count(trk_mag.array[this_app_slice]):
                hdg = trk_mag
                magnetic = True
            elif hdg_true and np.ma.count(hdg_true.array[this_app_slice]):
                hdg = hdg_true
                magnetic = False
            else:
                hdg = hdg_mag
                magnetic = True

            kwargs = {'runway': runway}

            if magnetic:
                try:
                    # If magnetic heading is being used get magnetic heading
                    # of runway
                    kwargs = {'heading': runway['magnetic_heading']}
                except KeyError:
                    # If magnetic heading is not know for runway fallback to
                    # true heading
                    pass

            # What is the heading with respect to the runway centreline for this approach?
            off_cl = runway_deviation(hdg.array[this_app_slice], **kwargs)
            off_cl = np.ma.filled(off_cl, fill_value=0.0)

            # Use recorded groundspeed where available, otherwise
            # estimate range using true airspeed. This is because there
            # are aircraft which record ILS but not groundspeed data. In
            # either case the speed is referenced to the runway heading
            # in case of large deviations on the approach or runway.
            if gspd:
                speed = gspd.array[this_app_slice] * \
                    np.cos(np.radians(off_cl))
                freq = gspd.frequency

            if not gspd or not np.ma.count(speed):
                speed = tas.array[this_app_slice] * \
                    np.cos(np.radians(off_cl))
                freq = tas.frequency

            # Estimate range by integrating back from zero at the end of the
            # phase to high range values at the start of the phase.
            spd_repaired = repair_mask(speed, repair_duration=None,
                                       extrapolate=True)
            app_range[this_app_slice] = integrate(spd_repaired, freq,
                                                  scale=ut.multiplier(ut.KT, ut.METER_S),
                                                  extend=True,
                                                  direction='reverse')

            _, app_slices = slices_between(alt_aal.array[this_app_slice],
                                           100, 500)
            # Computed locally, so app_slices do not need rescaling.
            if len(app_slices) != 1:
                self.info(
                    'Altitude AAL is not between 100-500 ft during an '
                    'approach slice. %s will not be calculated for this '
                    'section.', self.name)
                app_range[this_app_slice] = np_ma_masked_zeros_like(app_range[this_app_slice])
                continue

            # reg_slice is the slice of data over which we will apply a
            # regression process to identify the touchdown point from the
            # height and distance arrays.
            reg_slice = shift_slice(app_slices[0], this_app_slice.start)

            gs_est = approach.gs_est
            # Check we have enough valid glideslope data for the regression slice.
            if gs_est and np.ma.count(glide.array[reg_slice])>10:
                # Compute best fit glidepath. The term (1-0.13 x glideslope
                # deviation) caters for the aircraft deviating from the
                # planned flightpath. 1 dot low is about 7% of a 3 degree
                # glidepath. Not precise, but adequate accuracy for the small
                # error we are correcting for here, and empyrically checked.
                corr, slope, offset = coreg(app_range[reg_slice],
                    alt_aal.array[reg_slice] * (1 - 0.13 * glide.array[reg_slice]))
                # This should correlate very well, and any drop in this is a
                # sign of problems elsewhere.
                if corr < 0.995:
                    self.warning('Low convergence in computing ILS '
                                 'glideslope offset.')

                # We can be sure there is a glideslope antenna because we
                # captured the glidepath.
                try:
                    # Reference to the localizer as it is an ILS approach.
                    extend = runway_distances(runway)[1]  # gs_2_loc
                except (KeyError, TypeError):
                    # If ILS antennae coordinates not known, substitute the
                    # touchdown point 1000ft from start of runway
                    extend = runway_length(runway) - ut.convert(1000, ut.FT, ut.METER)

            else:
                # Just work off the height data assuming the pilot was aiming
                # to touchdown close to the glideslope antenna (for a visual
                # approach to an ILS-equipped runway) or at the touchdown
                # zone if no ILS glidepath is installed.
                corr, slope, offset = coreg(app_range[reg_slice],
                                            alt_aal.array[reg_slice])
                # This should still correlate pretty well, though not quite
                # as well as for a directed approach.
                if (corr or 0) < 0.990:
                    self.warning('Low convergence in computing visual '
                                 'approach path offset.')

                # If we have a glideslope antenna position, use this as the pilot will normally land abeam the antenna.
                try:
                    # Reference to the end of the runway as it is treated as a visual approach later on.
                    start_2_loc, gs_2_loc, end_2_loc, pgs_lat, pgs_lon = \
                        runway_distances(runway)
                    extend = gs_2_loc - end_2_loc
                except (KeyError, TypeError):
                    # If no ILS antennae, put the touchdown point 1000ft from start of runway
                    extend = runway_length(runway) - ut.convert(1000, ut.FT, ut.METER)


            # This plot code allows the actual flightpath and regression line
            # to be viewed in case of concern about the performance of the
            # algorithm.
            # x-reference set to 0=g/s position or aiming point.
            # blue = Altitude AAL
            # red = corrected for glideslope deviations
            # black = fitted slope.
            '''
            from analysis_engine.plot_flight import plot_parameter
            import matplotlib.pyplot as plt
            x1=app_range[gs_est.start:this_app_slice.stop] - offset
            y1=alt_aal.array[gs_est.start:this_app_slice.stop]
            x2=app_range[gs_est] - offset
            #
            y2=alt_aal.array[gs_est] * (1-0.13*glide.array[gs_est])
            xnew = np.linspace(np.min(x2),np.max(x2),num=2)
            ynew = (xnew)/slope
            plt.plot(x1,y1,'b-',x2,y2,'r-',xnew,ynew,'k-')
            plt.show()
            '''


            # Shift the values in this approach so that the range = 0 at
            # 0ft on the projected ILS or approach slope.
            app_range[this_app_slice] += extend - (offset or 0)

        self.array = app_range


##############################################################################


class VOR1Frequency(DerivedParameterNode):
    '''
    Extraction of VOR tuned frequencies from receiver (1).
    '''

    name = 'VOR (1) Frequency'
    units = ut.MHZ

    def derive(self, f=P('ILS-VOR (1) Frequency')):
        self.array = filter_vor_ils_frequencies(f.array, 'VOR')


class VOR2Frequency(DerivedParameterNode):
    '''
    Extraction of VOR tuned frequencies from receiver (1).
    '''

    name = 'VOR (2) Frequency'
    units = ut.MHZ

    def derive(self, f=P('ILS-VOR (2) Frequency')):
        self.array = filter_vor_ils_frequencies(f.array, 'VOR')

class WindSpeed(DerivedParameterNode):
    '''
    Required for Embraer 135-145 Data Frame
    '''

    align = False
    units = ut.KT

    def derive(self, wind_1=P('Wind Speed (1)'), wind_2=P('Wind Speed (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(wind_1, wind_2)


class WindDirection(DerivedParameterNode):
    '''
    Recorded wind direction is always True.

    Either combines two separate Wind Direction parameters.
    The Embraer 135-145 data frame includes two sources.
    '''

    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return (('Wind Direction (1)' in available or
                 'Wind Direction (2)' in available))

    def derive(self, wind_1=P('Wind Direction (1)'),
               wind_2=P('Wind Direction (2)')):

        if wind_1 or wind_2:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(wind_1, wind_2)


class WindDirectionTrue(DerivedParameterNode):
    '''
    This is a copy of the above parameter - Wind Direction.
    We need to keep that for now as some data exports use Wind Direction True.

    Previously this parameter was Wind Direction + magnetic variation, and was
    created in assumption that Wind Direction was magnetic, not true, which has
    been proven to be incorrect.
    '''

    align = False
    units = ut.DEGREE

    def derive(self, wind_dir=P('Wind Direction'),):

        self.array = wind_dir.array


class WindDirectionMagnetic(DerivedParameterNode):
    '''
    Compensates for magnetic variation, which will have been computed
    previously.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return 'Wind Direction' in available and \
               any_of(('Magnetic Variation From Runway', 'Magnetic Variation'),
                      available)

    def derive(self, wind=P('Wind Direction'),
               rwy_var=P('Magnetic Variation From Runway'),
               mag_var=P('Magnetic Variation')):
        var = mag_var.array if mag_var and np.ma.count(mag_var.array) else rwy_var.array
        self.array = (wind.array - var) % 360.0


class WheelSpeedLeft(DerivedParameterNode):
    '''
    Merge the various recorded wheel speed signals from the left hand bogie.
    '''

    name = 'Wheel Speed (L)'
    align = False
    units = ut.METER_S

    @classmethod
    def can_operate(cls, available):
        return 'Wheel Speed (L) (1)' in available

    def derive(self, ws_1=P('Wheel Speed (L) (1)'), ws_2=P('Wheel Speed (L) (2)'),
               ws_3=P('Wheel Speed (L) (3)'), ws_4=P('Wheel Speed (L) (4)')):
        sources = [ws_1, ws_2, ws_3, ws_4]
        self.offset = 0.0
        self.frequency = 4.0
        self.array = blend_parameters(sources, self.offset, self.frequency)


class WheelSpeedRight(DerivedParameterNode):
    '''
    Merge the various recorded wheel speed signals from the right hand bogie.
    '''

    name = 'Wheel Speed (R)'
    align = False
    units = ut.METER_S

    @classmethod
    def can_operate(cls, available):
        return 'Wheel Speed (R) (1)' in available

    def derive(self, ws_1=P('Wheel Speed (R) (1)'), ws_2=P('Wheel Speed (R) (2)'),
               ws_3=P('Wheel Speed (R) (3)'), ws_4=P('Wheel Speed (R) (4)')):
        sources = [ws_1, ws_2, ws_3, ws_4]
        self.offset = 0.0
        self.frequency = 4.0
        self.array = blend_parameters(sources, self.offset, self.frequency)


class AirspeedSelectedForApproaches(DerivedParameterNode):
    '''
    Use Airspeed Selected if frequency >= 0.25, otherwise upsample to 1Hz using
    next sampled value.
    '''
    units = ut.KT

    def derive(self, aspd=P('Airspeed Selected'), fast=S('Fast')):
        if aspd.frequency >= 0.25:
            self.array = aspd.array
            return

        rep = 1 // aspd.frequency
        array = repair_mask(mask_outside_slices(aspd.array, fast.get_slices()), method='fill_start', repair_duration=None)
        array = array.repeat(rep)
        if aspd.offset >= 1:
            # Compensate for the offset of the source parameter to align the
            # value steps with the recorded ones
            offset = int(aspd.offset)
            array = np.ma.concatenate(
                (np_ma_masked_zeros(offset), array[:-offset]))
        self.array = np.ma.concatenate([array[int(rep - 1):], array[-int((rep - 1)):]])
        self.frequency = 1
        self.offset = 0


class AirspeedSelected(DerivedParameterNode):
    '''
    Merge the various recorded Airspeed Selected signals.
    '''

    name = 'Airspeed Selected'
    align = False
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, as_l=P('Airspeed Selected (L)'),
               as_r=P('Airspeed Selected (R)'),
               as_mcp=P('Airspeed Selected (MCP)'),
               as_1=P('Airspeed Selected (1)'),
               as_2=P('Airspeed Selected (2)'),
               as_3=P('Airspeed Selected (3)'),
               as_4=P('Airspeed Selected (4)')):
        sources = [as_l, as_r, as_mcp, as_1, as_2, as_3, as_4]
        sources = [s for s in sources if s is not None]
        # Constrict number of sources to be a power of 2 for an even alignable
        # frequency.
        sources = sources[:power_floor(len(sources))]
        self.offset = 0.0
        self.frequency = len(sources) * sources[0].frequency
        self.array = blend_parameters(sources, self.offset, self.frequency)


class WheelSpeed(DerivedParameterNode):
    '''
    Merge Left and Right wheel speeds.
    '''
    # Q: Should wheel speed Centre (C) be merged too?
    align = False
    units = ut.METER_S

    def derive(self, ws_l=P('Wheel Speed (L)'), ws_r=P('Wheel Speed (R)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(ws_l, ws_r)


class Track(DerivedParameterNode):
    '''
    Magnetic Track Heading of the Aircraft by adding Drift from track to the
    aircraft Heading.

    Range 0 to 360
    '''

    units = ut.DEGREE

    def derive(self, heading=P('Heading'), drift=P('Drift')):
        self.array = (heading.array + drift.array) % 360.0


class TrackTrue(DerivedParameterNode):
    '''
    True Track Heading of the Aircraft by adding Drift from track to the
    aircraft's True Heading.

    Range 0 to 360
    '''

    units = ut.DEGREE

    def derive(self, heading=P('Heading True'), drift=P('Drift')):
        #Note: drift is to the right of heading, so: Track = Heading + Drift
        self.array = (heading.array + drift.array) % 360.0

class TrackContinuous(DerivedParameterNode):
    '''
    Magnetic Track Heading of the Aircraft by adding Drift from track to the
    aircraft Heading.

    Range is Continuous
    '''

    units = ut.DEGREE

    def derive(self, heading=P('Heading Continuous'), drift=P('Drift')):
        self.array = heading.array + drift.array


class TrackTrueContinuous(DerivedParameterNode):
    '''
    True Track Heading of the Aircraft by adding Drift from track to the
    aircraft's True Heading.

    Range is Continuous
    '''

    units = ut.DEGREE

    def derive(self, heading=P('Heading True Continuous'), drift=P('Drift')):
        #Note: drift is to the right of heading, so: Track = Heading + Drift
        self.array = heading.array + drift.array


class TrackDeviationFromRunway(DerivedParameterNode):
    '''
    Difference from the aircraft's Track angle and that of the Runway
    centreline. Measured during Takeoff and Approach phases.

    Based on Track True angle in preference to Track (magnetic) angle in
    order to avoid complications with magnetic deviation values recorded at
    airports. The deviation from runway centre line would be the same whether
    the calculation is based on Magnetic or True measurements.

    This parameter uses continous versions of Track and Track True as the input
    but returns a range from 0 to 360
    '''

    # force offset for approach slice start consistency
    align_frequency = 1
    align_offset = 0
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Approach Information', 'FDR Takeoff Runway'), available) \
               and any_of(('Track Continuous', 'Track True Continuous'), available)

    def _track_deviation(self, array, _slice, rwy, magnetic=False):
        if magnetic:
            try:
                # If magnetic heading is being used get magnetic heading
                # of runway
                self.array[_slice] = runway_deviation(
                     array[_slice], heading=rwy['magnetic_heading'])
                return
            except KeyError:
                # If magnetic heading is not known for runway fallback to
                # true heading
                self.warning('Runway magnetic heading not available; using True heading. Note: not accounting for magnetic variation!')
                pass
        try:
            self.array[_slice] = runway_deviation(array[_slice], runway=rwy)
        except ValueError:
            # could not determine runway information
            return

    def derive(self, track_true=P('Track True Continuous'),
               track_mag=P('Track Continuous'),
               takeoff=S('Takeoff Roll Or Rejected Takeoff'),
               to_rwy=A('FDR Takeoff Runway'),
               apps=App('Approach Information')):

        if track_true:
            magnetic = False
            track = track_true
        else:
            magnetic = True
            track = track_mag

        self.array = np_ma_masked_zeros_like(track.array)
        to_stop = None
        if to_rwy:
            # extend takeoff slice for 5 minutes after
            to_stop = takeoff[0].slice.stop+300
            _slice = slice(takeoff[0].slice.start, to_stop)
            self._track_deviation(track.array, _slice, to_rwy.value, magnetic)
        if apps:
            for app in apps:
                if app.type=='LANDING':
                    runway = app.landing_runway
                else:
                    runway = app.approach_runway
                if not runway:
                    self.warning("Cannot calculate TrackDeviationFromRunway for "
                                 "approach as there is no runway.")
                    continue
                # extend approach slice up to 15 minutes earlier (or to takeoff)
                # limit to previous approach stop
                prev_app = apps.get_previous(app.slice.start, use='start')
                prev_app_idx = prev_app.slice.stop if prev_app else 0
                idxs = [i for i in [to_stop, app.slice.start-900, prev_app_idx] if i]
                app_start = max(idxs) if idxs else None
                _slice = slice(app_start, app.slice.stop)
                self._track_deviation(track.array, _slice, runway, magnetic)


##############################################################################
# Velocity Speeds


########################################
# Takeoff Safety Speed (V2)


##class V2(DerivedParameterNode):
    ##'''
    ##Takeoff Safety Speed (V2) can be derived for different aircraft.

    ##If the value is provided in an achieved flight record (AFR), we use this in
    ##preference. This allows us to cater for operators that use improved
    ##performance tables so that they can provide the values that they used.

    ##For Airbus aircraft, if auto speed control is enabled, we can use the
    ##primary flight display selected speed value from the start of the takeoff
    ##run.

    ##Some other aircraft types record multiple parameters in the same location
    ##within data frames. We need to select only the data that we are interested
    ##in, i.e. the V2 values.

    ##The value is restricted to the range from the start of takeoff acceleration
    ##to the end of the initial climb flight phase.

    ##Reference was made to the following documentation to assist with the
    ##development of this algorithm:

    ##- A320 Flight Profile Specification
    ##- A321 Flight Profile Specification
    ##'''

    ##units = ut.KT

    ##@classmethod
    ##def can_operate(cls, available, afr_v2=A('AFR V2'),
                    ##manufacturer=A('Manufacturer')):

        ##afr = all_of((
            ##'Airspeed',
            ##'AFR V2',
            ##'Liftoff',
            ##'Climb Start',
        ##), available) and afr_v2 and afr_v2.value >= AIRSPEED_THRESHOLD

        ##airbus = all_of((
            ##'Airspeed',
            ##'Airspeed Selected',
            ##'Speed Control',
            ##'Liftoff',
            ##'Climb Start',
            ##'Manufacturer',
        ##), available) and manufacturer and manufacturer.value == 'Airbus'

        ##embraer = all_of((
            ##'Airspeed',
            ##'V2-Vac',
            ##'Liftoff',
            ##'Climb Start',
        ##), available)

        ##return afr or airbus or embraer

    ##def derive(self,
               ##airspeed=P('Airspeed'),
               ##v2_vac=A('V2-Vac'),
               ##spd_sel=P('Airspeed Selected'),
               ##spd_ctl=P('Speed Control'),
               ##afr_v2=A('AFR V2'),
               ##liftoffs=KTI('Liftoff'),
               ##climb_starts=KTI('Climb Start'),
               ##manufacturer=A('Manufacturer')):

        ### Prepare a zeroed, masked array based on the airspeed:
        ##self.array = np_ma_masked_zeros_like(airspeed.array, np.int)

        ### Determine interesting sections of flight which we want to use for V2.
        ### Due to issues with how data is recorded, use five superframes before
        ### liftoff until the start of the climb:
        ##starts = deepcopy(liftoffs)
        ##for start in starts:
            ##start.index = max(start.index - 5 * 64 * self.hz, 0)
        ##phases = slices_from_ktis(starts, climb_starts)

        ### 1. Use value provided in achieved flight record (if available):
        ##if afr_v2 and afr_v2.value >= AIRSPEED_THRESHOLD:
            ##for phase in phases:
                ##self.array[phase] = round(afr_v2.value)
            ##return

        ### 2. Derive parameter for Embraer 170/190:
        ##if v2_vac:
            ##for phase in phases:
                ##value = most_common_value(v2_vac.array[phase].astype(np.int))
                ##if value is not None:
                    ##self.array[phase] = value
            ##return

        ### 3. Derive parameter for Airbus:
        ##if manufacturer and manufacturer.value == 'Airbus':
            ##spd_sel.array[spd_ctl.array == 'Manual'] = np.ma.masked
            ##for phase in phases:
                ##value = most_common_value(spd_sel.array[phase].astype(np.int))
                ##if value is not None:
                    ##self.array[phase] = value
            ##return


class V2Lookup(DerivedParameterNode):
    '''
    Takeoff Safety Speed (V2) can be derived for different aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For V2, looking up a value requires the weight and flap (lever detents)
    at liftoff.

    Flap is used as the first dependency to avoid interpolation of flap detents
    when flap is recorded at a lower frequency than airspeed.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_series=A('Engine Series'), engine_type=A('Engine Type')):

        core = all_of((
            'Airspeed',
            'Liftoff',
            'Climb Start',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and lookup_table(cls, 'v2', *attrs)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airspeed=P('Airspeed'),
               weight_liftoffs=KPV('Gross Weight At Liftoff'),
               liftoffs=KTI('Liftoff'),
               climb_starts=KTI('Climb Start'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array, np.int)

        # Determine interesting sections of flight which we want to use for V2.
        # Due to issues with how data is recorded, use five superframes before
        # liftoff until the start of the climb:
        starts = deepcopy(liftoffs)
        for start in starts:
            start.index = max(start.index - 5 * 64 * self.hz, 0)
        phases = slices_from_ktis(starts, climb_starts)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'v2', *attrs)

        for phase in phases:

            if weight_liftoffs:
                weight_liftoff = weight_liftoffs.get_first(within_slice=phase)
                index, weight = weight_liftoff.index, weight_liftoff.value
            else:
                index, weight = liftoffs.get_first(within_slice=phase).index, None

            if index is None:
                continue

            detent = (flap_lever or flap_synth).array[int(index)]

            try:
                self.array[slices_int(phase)] = table.v2(detent, weight)
            except (KeyError, ValueError) as error:
                self.warning("Error in '%s': %s", self.name, error)
                # Where the aircraft takes off with flap settings outside the
                # documented V2 range, we need the program to continue without
                # raising an exception, so that the incorrect flap at takeoff
                # can be detected.
                continue


########################################
# Reference Speed (Vref)


class Vref(DerivedParameterNode):
    '''
    Reference Speed (Vref) can be derived for different aircraft.

    If the value is provided in an achieved flight record (AFR), we use this in
    preference. This allows us to cater for operators that use improved
    performance tables so that they can provide the values that they used.

    Some other aircraft types record multiple parameters in the same location
    within data frames. We need to select only the data that we are interested
    in, i.e. the Vref values.

    The value is restricted to the approach and landing phases which includes
    all approaches that result in landings and go-arounds.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, afr_vref=A('AFR Vref')):

        afr = all_of((
            'Airspeed',
            'AFR Vref',
            'Approach And Landing',
        ), available) and afr_vref and afr_vref.value >= AIRSPEED_THRESHOLD

        embraer = all_of((
            'Airspeed',
            'V1-Vref',
            'Approach And Landing',
        ), available)

        return afr or embraer

    def derive(self,
               airspeed=P('Airspeed'),
               v1_vref=P('V1-Vref'),
               afr_vref=A('AFR Vref'),
               approaches=S('Approach And Landing')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array, np.int)

        # 1. Use value provided in achieved flight record (if available):
        if afr_vref and afr_vref.value >= AIRSPEED_THRESHOLD:
            for approach in approaches:
                self.array[slices_int(approach.slice)] = round(afr_vref.value)
            return

        # 2. Derive parameter for Embraer 170/190:
        if v1_vref:
            for approach in approaches:
                value = most_common_value(v1_vref.array[slices_int(approach.slice)].astype(np.int))
                if value is not None:
                    self.array[slices_int(approach.slice)] = value
            return


class VrefLookup(DerivedParameterNode):
    '''
    Reference Speed (Vref) can be derived for different aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For Vref, looking up a value requires the weight and flap (lever detents)
    at touchdown or the lowest point in a go-around.

    Flap is used as the first dependency to avoid interpolation of flap detents
    when flap is recorded at a lower frequency than airspeed.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        core = all_of((
            'Airspeed',
            'Approach And Landing',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        weight = any_of((
            'Gross Weight Smoothed',
            'Touchdown',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and weight and lookup_table(cls, 'vref', *attrs)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               air_spd=P('Airspeed'),
               gw=P('Gross Weight Smoothed'),
               approaches=S('Approach And Landing'),
               touchdowns=KTI('Touchdown'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(air_spd.array, np.int)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vref', *attrs)

        # If we have gross weight, repair gaps up to 2 superframes in length:
        if gw is not None:
            try:
                # TODO: Things to consider related to gross weight:
                #       - Does smoothed gross weight need to be repaired?
                repaired_gw = repair_mask(gw.array, repair_duration=130,
                                          extrapolate=True)
            except ValueError:
                self.warning("'%s' will be fully masked because '%s' array "
                             "could not be repaired.", self.name, gw.name)
                return

        # Determine the maximum detent in advance to avoid multiple lookups:
        parameter = flap_lever or flap_synth
        max_detent = max(table.vref_detents, key=lambda x: parameter.state.get(x, -1))

        for approach in approaches:
            phase = slices_int(approach.slice)
            # Select the maximum flap detent during the phase:
            index, detent = max_value(parameter.array, phase)
            # Allow no gross weight for aircraft which use a fixed vspeed:
            weight = repaired_gw[int(index)] if gw is not None else None

            if touchdowns.get(within_slice=phase) or detent in table.vref_detents:
                # We either touched down, so use the touchdown flap lever
                # detent, or we had reached a maximum flap lever detent during
                # the approach which is in the vref table.
                pass
            else:
                # Not the final landing and max detent not in vspeed table,
                # so use the maximum detent possible as a reference.
                self.info("No touchdown in this approach and maximum "
                          "%s '%s' not in lookup table. Using max "
                          "possible detent '%s' as reference.",
                          parameter.name, detent, max_detent)
                detent = max_detent

            try:
                self.array[phase] = table.vref(detent, weight)
            except (KeyError, ValueError) as error:
                self.warning("Error in '%s': %s", self.name, error)
                # Where the aircraft takes off with flap settings outside the
                # documented vref range, we need the program to continue without
                # raising an exception, so that the incorrect flap at landing
                # can be detected.
                continue


########################################
# Approach Speed (Vapp)


class Vapp(DerivedParameterNode):
    '''
    Approach Speed (Vapp) can be derived for different aircraft.

    If the value is provided in an achieved flight record (AFR), we use this in
    preference. This allows us to cater for operators that use improved
    performance tables so that they can provide the values that they used.

    Some other aircraft types record multiple parameters in the same location
    within data frames. We need to select only the data that we are interested
    in, i.e. the Vapp values.

    The value is restricted to the approach and landing phases which includes
    all approaches that result in landings and go-arounds.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, afr_vapp=A('AFR Vapp')):

        afr = all_of((
            'Airspeed',
            'AFR Vapp',
            'Approach And Landing',
        ), available) and afr_vapp and afr_vapp.value >= AIRSPEED_THRESHOLD

        embraer = all_of((
            'Airspeed',
            'VR-Vapp',
            'Approach And Landing',
        ), available)

        return afr or embraer

    def derive(self,
               airspeed=P('Airspeed'),
               vr_vapp=A('VR-Vapp'),
               afr_vapp=A('AFR Vapp'),
               approaches=S('Approach And Landing')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array, np.int)

        # 1. Use value provided in achieved flight record (if available):
        if afr_vapp and afr_vapp.value >= AIRSPEED_THRESHOLD:
            for approach in approaches:
                self.array[slices_int(approach.slice)] = round(afr_vapp.value)
            return

        # 2. Derive parameter for Embraer 170/190:
        if vr_vapp:
            for approach in approaches:
                value = most_common_value(vr_vapp.array[slices_int(approach.slice)].astype(np.int))
                if value is not None:
                    self.array[slices_int(approach.slice)] = value
            return


class VappLookup(DerivedParameterNode):
    '''
    Approach Speed (Vapp) can be derived for different aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For Vapp, looking up a value requires the weight and flap (lever detents)
    at touchdown or the lowest point in a go-around.

    Flap is used as the first dependency to avoid interpolation of flap detents
    when flap is recorded at a lower frequency than airspeed.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        core = all_of((
            'Airspeed',
            'Approach And Landing',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        weight = any_of((
            'Gross Weight Smoothed',
            'Touchdown',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and weight and lookup_table(cls, 'vapp', *attrs)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               air_spd=P('Airspeed'),
               gw=P('Gross Weight Smoothed'),
               approaches=S('Approach And Landing'),
               touchdowns=KTI('Touchdown'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(air_spd.array, np.int)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vapp', *attrs)

        # If we have gross weight, repair gaps up to 2 superframes in length:
        if gw is not None:
            try:
                # TODO: Things to consider related to gross weight:
                #       - Does smoothed gross weight need to be repaired?
                repaired_gw = repair_mask(gw.array, repair_duration=130,
                                          extrapolate=True)
            except ValueError:
                self.warning("'%s' will be fully masked because '%s' array "
                             "could not be repaired.", self.name, gw.name)
                return

        # Determine the maximum detent in advance to avoid multiple lookups:
        parameter = flap_lever or flap_synth
        max_detent = max(table.vapp_detents, key=lambda x: parameter.state.get(x, -1))

        for approach in approaches:
            phase = slices_int(approach.slice)
            # Select the maximum flap detent during the phase:
            index, detent = max_value(parameter.array, phase)
            # Allow no gross weight for aircraft which use a fixed vspeed:
            weight = repaired_gw[index] if gw is not None else None

            if touchdowns.get(within_slice=phase) or detent in table.vapp_detents:
                # We either touched down, so use the touchdown flap lever
                # detent, or we had reached a maximum flap lever detent during
                # the approach which is in the vapp table.
                pass
            else:
                # Not the final landing and max detent not in vspeed table,
                # so use the maximum detent possible as a reference.
                self.info("No touchdown in this approach and maximum "
                          "%s '%s' not in lookup table. Using max "
                          "possible detent '%s' as reference.",
                          parameter.name, detent, max_detent)
                detent = max_detent

            try:
                self.array[phase] = table.vapp(detent, weight)
            except (KeyError, ValueError) as error:
                self.warning("Error in '%s': %s", self.name, error)
                # Where the aircraft takes off with flap settings outside the
                # documented vapp range, we need the program to continue without
                # raising an exception, so that the incorrect flap at landing
                # can be detected.
                continue


########################################
# Lowest Selectable Speed (VLS)

class VLSLookup(DerivedParameterNode):
    '''
    Lowest Selectable Speed (VLS) can be derived for Airbus aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For VLS, looking up a value requires configuration and in some cases CG.
    '''

    name = 'VLS Lookup'
    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        if family and not family.value in ('A319', 'A320', 'A321', 'A330', 'A340'):
            return False

        core = all_of((
            'Airspeed',
            'Approach And Landing',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
            'Gross Weight Smoothed',
            'Airborne',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and lookup_table(cls, 'vls', *attrs)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               air_spd=P('Airspeed'),
               gw=P('Gross Weight Smoothed'),
               approaches=S('Approach And Landing'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series'),
               center_of_gravity=P('Center Of Gravity'),
               alt_std=P('Altitude STD Smoothed'),
               airborne=S('Airborne')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(air_spd.array, np.int)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vls', *attrs)

        # Repair gaps in Gross Weight up to 2 superframes in length:
        try:
            repaired_gw = repair_mask(gw.array, repair_duration=130,
                                      extrapolate=True)
        except ValueError:
            self.warning("'%s' will be fully masked because '%s' array "
                         "could not be repaired.", self.name, gw.name)
            return

        parameter = flap_lever or flap_synth

        for approach in approaches:
            phase = slices_int(approach.slice)
            # Find the maximum flap detent selection point during the phase:
            max_flap_selected = max_value(parameter.array, phase)
            if max_flap_selected.index:
                weight = repaired_gw[max_flap_selected.index]
            else:
                weight = None
            if center_of_gravity:
                cg = center_of_gravity.array[max_flap_selected.index]
            else:
                cg = None

            try:
                for state in (parameter.array.values_mapping[r] for r in np.ma.unique(parameter.array.raw.compressed())):
                    self.array[parameter.array == state] = table.vls(state, weight, cg)
            except (KeyError, ValueError) as error:
                self.warning("Error in '%s': %s", self.name, error)
                continue

        # Compute VLS clean
        table = lookup_table(self, 'vls_clean', *attrs)

        if table is None or alt_std is None:
            return

        # Repair gaps in Altitude STD Smoothed up to 2 superframes in length:
        try:
            repaired_alt = repair_mask(alt_std.array, repair_duration=130,
                                          extrapolate=True)
        except ValueError:
            self.warning("'%s' will be fully masked for VLS clean because "
                         "'%s' array could not be repaired.", self.name, alt_std.name)
            return

        flaps_0 = parameter.array == 'Lever 0'
        self.array[flaps_0] = table.vls_clean(repaired_gw[flaps_0], repaired_alt[flaps_0])

        # We want to mask out grounded sections of flight:
        self.array = mask_outside_slices(self.array, airborne.get_slices())



########################################
# Maximum Operating Speed (VMO)


class VMOLookup(DerivedParameterNode):
    '''
    Maximum Operating Speed (VMO) can be derived for different aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For VMO, looking up a value requires the pressure altitude.
    '''

    name = 'VMO Lookup'
    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        core = all_of((
            'Altitude STD Smoothed',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and lookup_table(cls, 'vmo', *attrs)

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vmo', *attrs)

        self.array = table.vmo(alt_std.array)


########################################
# Maximum Operating Mach (MMO)


class MMOLookup(DerivedParameterNode):
    '''
    Maximum Operating Mach (MMO) can be derived for different aircraft.

    In cases where values cannot be derived solely from recorded parameters, we
    can make use of a look-up table to determine values for velocity speeds.

    For MMO, looking up a value requires the pressure altitude.
    '''

    name = 'MMO Lookup'
    units = ut.MACH

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        core = all_of((
            'Altitude STD Smoothed',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and lookup_table(cls, 'mmo', *attrs)

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'mmo', *attrs)

        self.array = table.mmo(alt_std.array)


########################################
# Minimum Airspeed


class MinimumAirspeed(DerivedParameterNode):
    '''
    Minimum airspeed at which there is suitable manoeuvrability.

    For Boeing aircraft, use the minimum manoeuvre speed or the minimum
    operating speed depending on availability.

    For Airbus aircraft, use the lowest selectable airspeed (VLS).

    - Airbus Flight Crew Operating Manual (FCOM) (For All Types)
    - Boeing Flight Crew Training Manual (FCTM) (For All Types)
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        core = all_of(('Airborne', 'Airspeed'), available)
        a = any_of((
            'FC Min Operating Speed',
            'Min Operating Speed',
            'VLS',
            'VLS Lookup',
        ), available)
        b = any_of((
            'Minimum Clean Lookup',
        ), available)
        f = any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available)
        return core and (a or (b and f))

    def derive(self,
               airspeed=P('Airspeed'),
               mos_fc=P('FC Min Operating Speed'),
               mos=P('Min Operating Speed'),
               vls=P('VLS'),
               vls_lookup=P('VLS Lookup'),
               min_clean=P('Minimum Clean Lookup'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        # Use whatever minimum speed parameter we have available:
        parameter = first_valid_parameter(vls, vls_lookup, min_clean, mos_fc, mos)
        if not parameter:
            self.array = np_ma_masked_zeros_like(airspeed.array)
            return
        else:
            self.array = parameter.array

        # Handle where minimum manoeuvre speed is for clean configuration only:
        if parameter is min_clean:
            flap = flap_lever or flap_synth
            flaps_up = flap.array == '0'
            self.array[~flaps_up] = np.ma.masked
            self.array[flaps_up] = min_clean.array[flaps_up]

        # We want to mask out grounded sections of flight:
        self.array = mask_outside_slices(self.array, airborne.get_slices())


class MinimumCleanLookup(DerivedParameterNode):
    '''
    Minimum Clean Speed Lookup

    757/767: Vref30+80kts below FL250, Vref30+100kts above FL250
    '''
    units = ut.KT

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        return family and family.value in ('B757', 'B767') and \
               all_of(cls.get_dependency_names(), available)

    def derive(self,
               air_spd=P('Airspeed'),
               gw=P('Gross Weight Smoothed'),
               airborne=S('Airborne'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series'),
               alt_std=P('Altitude STD Smoothed'),
               crz=S('Cruise'),):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(air_spd.array, np.int)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vref', *attrs)

        # Determine the sections of flight to populate:
        phases = airborne.get_slices()

        # Need Vref30, so detent is pre set.
        detent = '30'

        for phase in phases:
            self.array[slices_int(phase)] = table.vref(detent, gw.array[slices_int(phase)])
            # Add 80kts to the whole array to get Vref30+80kts
            self.array[slices_int(phase)] += 80

        # above_FL250 includes S('Cruise') slices above FL247 in order to avoid
        # creating 'spikes' when the cruising altitude is FL250
        above_FL250 = slices_or(alt_std.slices_above(25000),
                                slices_and(crz.get_slices(), alt_std.slices_above(24700)))

        # Add 20kts to get Vref30+100kts above FL250
        for section in above_FL250:
            self.array[slices_int(section)] += 20


########################################
# Flap Manoeuvre Speed


class FlapManoeuvreSpeed(DerivedParameterNode):
    '''
    Flap manoeuvring speed for various flap settings.

    The flap manoeuvring speed guarantees full manoeuvre capability or at least
    a certain number of degrees of bank to stick shaker within a few thousand
    feet of the airport altitude.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - Boeing Flight Crew Training Manual (FCTM) (For All Types)
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer'),
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):

        if not manufacturer or not manufacturer.value == 'Boeing':
            return False

        try:
            at.get_fms_map(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No flap manoeuvre speed tables available for '%s', "
                        "'%s', '%s'.", model.value, series.value, family.value)
            return False

        core = all_of((
            'Airspeed', 'Altitude STD Smoothed',
            'Gross Weight Smoothed', 'Model', 'Series', 'Family',
            'Engine Type', 'Engine Series',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and lookup_table(cls, 'vref', *attrs)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               gw=P('Gross Weight Smoothed'),
               vref_25=P('Vref (25)'),
               vref_30=P('Vref (30)'),
               alt_std=P('Altitude STD Smoothed'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):

        flap = flap_lever or flap_synth

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vref', *attrs)

        # Lookup the table for recommended flap manoeuvring speeds:
        fms_table = at.get_fms_map(model.value, series.value, family.value)

        # For each flap detent calculate the flap manoeuvring speed:
        for detent, slices in slices_of_runs(flap.array):

            fms = fms_table.get(detent)
            if fms is None:
                continue  # skip to next detent as no value is defined.
            elif isinstance(fms[0], tuple):
                for weight, speed in reversed(fms):
                    condition = runs_of_ones(gw.array <= weight)
                    for s in slices_and(slices, condition):
                        self.array[s] = speed
            elif isinstance(fms[0], six.string_types):
                setting, offset = fms
                vref_recorded = locals().get('vref_%s' % setting)
                for s in slices:
                    # Use recorded vref if available else use lookup tables:
                    if vref_recorded is not None:
                        vref = vref_recorded.array[s]
                    elif gw.array[s].mask.all():
                        continue  # If the slice is all masked, skip...
                    else:
                        vref = table.vref(setting, gw.array[s])
                    self.array[s] = vref + offset
            else:
                raise TypeError('Encountered invalid table.')

        # We want to mask out sections of flight where the aircraft is not
        # airborne and where the aircraft is above 20000ft as, for the majority
        # of aircraft, flaps should not be extended above that limit.
        phases = slices_and(alt_std.slices_below(20000), alt_std.slices_above(50))
        self.array = mask_outside_slices(self.array, phases)


##############################################################################
# Relative Airspeeds


class AirspeedMinusAirspeedSelectedFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to Airspeed Selected During Approach which in case of low
    sample rate uses next sample of Airspeed Selected.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed', 'Airspeed Selected For Approaches'), available)

    def derive(self, aspd=P('Airspeed'), aspd_sel=P('Airspeed Selected For Approaches')):
        self.array = second_window(aspd.array - aspd_sel.array,
                                   self.frequency, 3)


########################################
# Airspeed Minus Airspeed Selected (FMS)

class AirspeedMinusAirspeedSelectedFMS(DerivedParameterNode):
    '''
    Airspeed relative to Airspeed Selected (FMS) during approach.
    '''
    units = ut.KT
    name = 'Airspeed Minus Airspeed Selected (FMS)'

    def derive(self,
               airspeed=P('Airspeed'),
               fms=P('Airspeed Selected (FMS)'),
               approaches=S('Approach And Landing')):
        # Prepare a zored, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        for approach in approaches:
            phase = slices_int(approach.slice)
            self.array[phase] = airspeed.array[phase] - fms.array[phase]


class AirspeedMinusAirspeedSelectedFMSFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to Airspeed Selected (FMS) over a 3 second window.
    '''
    align_frequency = 2
    align_offset = 0
    units = ut.KT
    name = 'Airspeed Minus Airspeed Selected (FMS) For 3 Sec'

    def derive(self, speed=P('Airspeed Minus Airspeed Selected (FMS)')):
        self.array = second_window(speed.array, self.frequency, 3)


########################################
# Airspeed Minus V2


class AirspeedMinusV2(DerivedParameterNode):
    '''
    Airspeed relative to the Takeoff Safety Speed (V2).

    Values of V2 are taken from recorded or derived values if available,
    otherwise we fall back to using a value from a lookup table.

    We also check to ensure that we have some valid samples in any recorded or
    derived parameter, otherwise, again, we fall back to lookup tables. To
    avoid issues with small samples of invalid data, we check that the area of
    data we are interested in has no masked values.

    As an additional step for the V2 parameter, we repair the masked values and
    extrapolate as sometimes the recorded parameter does not extend beyond the
    period during which the aircraft is on the runway.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return (all_of(('Airspeed', 'Liftoff', 'Climb Start',
                        'Climb Acceleration Start', 'Grounded'), available) and
                any_of(('V2 At Liftoff', 'Airspeed Selected At Takeoff Acceleration Start',
                        'V2 Lookup At Liftoff'), available))

    def derive(self,
               airspeed=P('Airspeed'),
               v2_recorded=KPV('V2 At Liftoff'),
               airspeed_selected=KPV('Airspeed Selected At Takeoff Acceleration Start'),
               v2_lookup=KPV('V2 Lookup At Liftoff'),
               liftoffs=KTI('Liftoff'),
               climb_starts=KTI('Climb Start'),
               climb_accel_starts=KTI('Climb Acceleration Start'),
               grounded=S('Grounded')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        # Determine interesting sections of flight which we want to use for V2.
        # Due to issues with how data is recorded, use five superframes before
        # liftoff until the start of the climb:
        phases = []
        for liftoff in liftoffs:
            ground = grounded.get_previous(liftoff.index, use='start')
            search_start = max(liftoff.index - 5 * 64 * self.hz, 0)
            # Avoid touchdown at touch and go which should be relative to vref/vapp
            start_index = max(search_start, ground.slice.start + 1 if ground else 0)
            my_climb_start = climb_starts.get_next(liftoff.index)
            if my_climb_start:
                # Extend to the greatest of Climb Start or Climb Acceleration Start
                my_climb_accel_start = climb_accel_starts.get_next(liftoff.index)
                stop_index = max(my_climb_start.index, my_climb_accel_start.index if my_climb_accel_start else 0)
            else:
                continue
            phases.append((search_start, start_index, stop_index))

        v2 = v2_recorded or airspeed_selected or v2_lookup
        if not v2:
            return

        for search_start, start_index, stop_index in phases:
            my_v2 = v2.get_last(within_slice=slice(search_start, stop_index+1))
            if my_v2 is not None and my_v2.value is not None:
                phase = slices_int(start_index, stop_index+1)
                self.array[phase] = airspeed.array[phase] - my_v2.value


class AirspeedMinusV2For3Sec(DerivedParameterNode):
    '''
    Airspeed relative to the Takeoff Safety Speed (V2) over a 3 second window.

    See the derived parameter 'Airspeed Minus V2' for further details.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self, speed=P('Airspeed Minus V2')):

        self.array = second_window(speed.array, self.frequency, 3)


########################################
# Airspeed Minus Vref


class AirspeedMinusVref(DerivedParameterNode):
    '''
    Airspeed relative to the Reference Speed (Vref).

    Values of Vref are taken from recorded or derived values if available,
    otherwise we fall back to using a value from a lookup table.

    We also check to ensure that we have some valid samples in any recorded or
    derived parameter, otherwise, again, we fall back to lookup tables. To
    avoid issues with small samples of invalid data, we check that the area of
    data we are interested in has no masked values.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return all_of((
            'Airspeed',
            'Approach And Landing',
        ), available) and any_of(('Vref', 'Vref Lookup'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               vref_recorded=P('Vref'),
               vref_lookup=P('Vref Lookup'),
               approaches=S('Approach And Landing')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        phases = slices_int(approaches.get_slices())

        vref = first_valid_parameter(vref_recorded, vref_lookup, phases=phases)

        if vref is None:
            return

        for phase in phases:
            value = most_common_value(vref.array[phase].astype(np.int))
            if value is not None:
                self.array[phase] = airspeed.array[phase] - value


class AirspeedMinusVrefFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to the Reference Speed (Vref) over a 3 second window.

    See the derived parameter 'Airspeed Minus Vref' for further details.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self, speed=P('Airspeed Minus Vref')):

        self.array = second_window(speed.array, self.frequency, 3)


########################################
# Airspeed Minus Vapp


class AirspeedMinusVapp(DerivedParameterNode):
    '''
    Airspeed relative to the Approach Speed (Vapp).

    Values of Vapp are taken from recorded or derived values if available,
    otherwise we fall back to using a value from a lookup table.

    We also check to ensure that we have some valid samples in any recorded or
    derived parameter, otherwise, again, we fall back to lookup tables. To
    avoid issues with small samples of invalid data, we check that the area of
    data we are interested in has no masked values.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return all_of((
            'Airspeed',
            'Approach And Landing',
        ), available) and any_of(('Vapp', 'Vapp Lookup'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               vapp_recorded=P('Vapp'),
               vapp_lookup=P('Vapp Lookup'),
               approaches=S('Approach And Landing')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        phases = slices_int(approaches.get_slices())

        vapp = first_valid_parameter(vapp_recorded, vapp_lookup, phases=phases)

        if vapp is None:
            return

        for phase in phases:
            if vapp.name == 'Vapp':
                # we have the recorded or value provided in derived parameter
                # from AFR field, so we can use the entire array
                self.array[phase] = airspeed.array[phase] - vapp.array[phase]
            else:
                # we have the lookup parameter
                value = most_common_value(vapp.array[phase].astype(np.int))
                if value is None:
                    continue
                self.array[phase] = airspeed.array[phase] - value


class AirspeedMinusVappFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to the Approach Speed (Vapp) over a 3 second window.

    See the derived parameter 'Airspeed Minus Vapp' for further details.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self, speed=P('Airspeed Minus Vapp')):

        self.array = second_window(speed.array, self.frequency, 3)

########################################
# Airspeed Minus VLS


class AirspeedMinusVLS(DerivedParameterNode):
    '''
    Airspeed relative to VLS.
    '''

    name = 'Airspeed Minus VLS'
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return all_of((
            'Airspeed',
            'Approach And Landing',
        ), available) and any_of(('VLS', 'VLS Lookup'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               vls_recorded=P('VLS'),
               vls_lookup=P('VLS Lookup'),
               approaches=S('Approach And Landing')):

        # Prepare a zeroed, masked array based on the airspeed:
        self.array = np_ma_masked_zeros_like(airspeed.array)

        # Determine the sections of flight where data must be valid:
        phases = slices_int(approaches.get_slices())

        # Using first_valid_parameter so that once lookup tables are introduced we'll just add it here
        vls = first_valid_parameter(vls_recorded, vls_lookup, phases=phases)

        if vls is None:
            return

        for phase in phases:
            self.array[phase] = airspeed.array[phase] - vls.array[phase]


class AirspeedMinusVLSFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to VLS over a 3 second window.

    See the derived parameter 'Airspeed Minus VLS' for further details.
    '''

    name = 'Airspeed Minus VLS For 3 Sec'
    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self, speed=P('Airspeed Minus VLS')):

        self.array = second_window(speed.array, self.frequency, 3)


########################################
# Airspeed Minus Minimum Airspeed


class AirspeedMinusMinimumAirspeed(DerivedParameterNode):
    '''
    Airspeed relative to minimum airspeed.

    See the derived parameter 'Minimum Airspeed' for further details.
    '''

    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed'),
               minimum_airspeed=P('Minimum Airspeed')):

        self.array = airspeed.array - minimum_airspeed.array


class AirspeedMinusMinimumAirspeedFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to minimum airspeed over a 3 second window.

    See the derived parameter 'Airspeed Minus Minimum Airspeed' for further
    details.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self, speed=P('Airspeed Minus Minimum Airspeed')):

        self.array = second_window(speed.array, self.frequency, 3)


########################################
# Airspeed Minus Flap Manoeuvre Speed


class AirspeedMinusFlapManoeuvreSpeed(DerivedParameterNode):
    '''
    Airspeed relative to flap manoeuvre speeds.

    See the derived parameter 'Flap Manoeuvre Speed' for further details.
    '''

    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed'),
               fms=P('Flap Manoeuvre Speed')):

        self.array = airspeed.array - fms.array



class AirspeedMinusFlapManoeuvreSpeedFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to flap manoeuvre speed over a 3 second window.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed'),
               fms=P('Flap Manoeuvre Speed')):

        speed = airspeed.array - fms.array

        self.array = second_window(speed, self.frequency, 3, extend_window=True)

########################################
# Airspeed Relative


class AirspeedRelative(DerivedParameterNode):
    '''
    Airspeed relative to Vref/Vapp.

    See the derived parameters 'Airspeed Minus Vref' and 'Airspeed Minus Vapp'
    for further details.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Airspeed Minus V2',
            'Airspeed Minus Vapp',
            'Airspeed Minus Vref',
        ), available)

    def derive(self,
               takeoff=P('Airspeed Minus V2'),
               vapp=P('Airspeed Minus Vapp'),
               vref=P('Airspeed Minus Vref')):

        approach = vapp or vref
        app_array = approach.array if approach else np_ma_masked_zeros_like(takeoff.array)

        if takeoff:
            toff_array = takeoff.array
            # We know the two areas of interest cannot overlap so we just add
            # the values, inverting the mask to provide a 0=ignore, 1=use_this
            # multiplier.
            speeds = toff_array.data*~toff_array.mask + app_array.data*~app_array.mask
            # And build this back into an array, masked only where both were
            # masked.
            combined = np.ma.array(data=speeds,
                                   mask = np.logical_and(toff_array.mask, app_array.mask))
            self.array = combined
        else:
            self.array = app_array


class AirspeedRelativeFor3Sec(DerivedParameterNode):
    '''
    Airspeed relative to Vapp/Vref over a 3 second window.

    See the derived parameter 'Airspeed Relative' for further details.
    '''

    align_frequency = 2
    align_offset = 0
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Airspeed Minus V2 For 3 Sec',
            'Airspeed Minus Vapp For 3 Sec',
            'Airspeed Minus Vref For 3 Sec',
        ), available)

    def derive(self,
               takeoff=P('Airspeed Minus V2 For 3 Sec'),
               vapp=P('Airspeed Minus Vapp For 3 Sec'),
               vref=P('Airspeed Minus Vref For 3 Sec')):

        approach = vapp or vref
        app_array = approach.array if approach else np_ma_masked_zeros_like(takeoff.array)
        if takeoff:
            toff_array = takeoff.array
            # We know the two areas of interest cannot overlap so we just add
            # the values, inverting the mask to provide a 0=ignore, 1=use_this
            # multiplier.
            speeds = toff_array.data*~toff_array.mask + app_array.data*~app_array.mask
            # And build this back into an array, masked only where both were
            # masked.
            combined = np.ma.array(data=speeds,
                                   mask = np.logical_and(toff_array.mask, app_array.mask))
            self.array = combined
        else:
            self.array = app_array


########################################
# Aircraft Energy

class KineticEnergy(DerivedParameterNode):
    '''Caclculate the kinetic energy of Aircraft in MegaJoule'''

    units = ut.MJ

    def derive(self,airspeed=P('Airspeed True'),
               mass=P('Gross Weight Smoothed')):
        v = ut.convert(airspeed.array, ut.KT, ut.METER_S)
        # m is in kg
        self.array = (0.5 * mass.array * v ** 2 * 10 **-6) # converted to MJoule


class PotentialEnergy(DerivedParameterNode):
    '''Potential energy of Aircraft'''

    align_frequency = 1
    units = ut.MJ

    def derive(self, altitude_aal=P('Altitude AAL'),
               gross_weight_smoothed=P('Gross Weight Smoothed')):
        altitude = altitude_aal.array * ut.CONVERSION_MULTIPLIERS[ut.FT][ut.METER]
        self.array = gross_weight_smoothed.array * GRAVITY_METRIC * altitude / 10 ** 6


class AircraftEnergy(DerivedParameterNode):
    '''Total energy of Aircraft'''

    units = ut.MJ

    def derive(self, potential_energy=P('Potential Energy'),
               kinetic_energy=P('Kinetic Energy')):
        self.array = potential_energy.array + kinetic_energy.array

