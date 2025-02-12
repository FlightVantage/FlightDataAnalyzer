# -*- coding: utf-8 -*-

import numpy as np

from flightdatautilities import units as ut

from analysis_engine.library import (
    any_deps,
    any_of,
    blend_parameters,
    blend_two_parameters,
    np_ma_masked_zeros_like,
    repair_mask,
    straighten_longitude,
)
from analysis_engine.node import DerivedParameterNode, P, A
from analysis_engine.derived_parameters import CoordinatesStraighten


class Groundspeed(DerivedParameterNode):

    align = False
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return any_deps(cls, available)

    def derive(self,
               # aeroplane
               source_A=P('Groundspeed (1)'),
               source_B=P('Groundspeed (2)')):
        self.array, self.frequency, self.offset = blend_two_parameters(
            source_A, source_B
        )

class LongitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    See Latitude Smoothed for notes.
    """
    name = 'Longitude Prepared'
    align = False
    units = ut.DEGREE

    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon=P('Longitude'), lat=P('Latitude'),
               ac_type=A('Aircraft Type')):
        """
        This removes the jumps in longitude arising from the poor resolution of
        the recorded signal.
        """
        self.array = self._smooth_coordinates(lon, lat, ac_type)

class LatitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    Creates Latitude Prepared from smoothed Latitude and Longitude parameters.
    See Latitude Smoothed for notes.
    """
    name = 'Latitude Prepared'
    align = False
    units = ut.DEGREE

    # Note force to 1Hz operation as latitude & longitude can be only
    # recorded at 0.25Hz.
    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon=P('Longitude'),
               lat=P('Latitude'),
               ac_type=A('Aircraft Type')):
        self.array = self._smooth_coordinates(lat, lon, ac_type)


class Latitude(DerivedParameterNode):
    """
    Blending the Latitude parameters
    """

    name = 'Latitude'
    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Latitude (1)', 'Latitude (2)', 'Latitude (3)'),
                      available)

    def derive(self,
               src_1=P('Latitude (1)'),
               src_2=P('Latitude (2)'),
               src_3=P('Latitude (3)')):

        sources = [
            source for source in [src_1, src_2, src_3] if source is not None \
            and np.count_nonzero(source.array) > len(source.array)/2
        ]

        if len(sources) == 1:
            self.offset = sources[0].offset
            self.frequency = sources[0].frequency
            self.array = sources[0].array

        elif len(sources) == 2:
            self.array, self.frequency, self.offset = blend_two_parameters(
                sources[0], sources[1]
            )

        elif len(sources) > 2:
            self.offset = 0.0
            self.frequency = 8.0
            self.array = blend_parameters(sources, offset=self.offset,
                                          frequency=self.frequency)
        else:
            self.array = np_ma_masked_zeros_like(src_1.array)


class Longitude(DerivedParameterNode):
    """
    Blending the Longitude parameters
    """

    name = 'Longitude'
    align = False
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Longitude (1)', 'Longitude (2)', 'Longitude (3)'),
                      available)

    def derive(self,
               src_1=P('Longitude (1)'),
               src_2=P('Longitude (2)'),
               src_3=P('Longitude (3)')):

        sources = [
            source for source in [src_1, src_2, src_3] if source is not None \
            and np.count_nonzero(source.array) > len(source.array)/2
        ]
        if len(sources) > 1:
            for source in sources:
                source.array = repair_mask(
                    straighten_longitude(source.array) + 180.0
                )

        if len(sources) == 1:
            self.offset = sources[0].offset
            self.frequency = sources[0].frequency
            self.array = sources[0].array

        elif len(sources) == 2:
            blended, self.frequency, self.offset = blend_two_parameters(
                sources[0], sources[1]
            )
            self.array = blended % 360 - 180.0

        elif len(sources) > 2:
            self.offset = 0.0
            self.frequency = 8.0
            blended = blend_parameters(sources, offset=self.offset,
                                       frequency=self.frequency)
            self.array = blended % 360 - 180.0

        else:
            self.array = np_ma_masked_zeros_like(src_1.array)
