import itertools
import os
import numpy as np
import sys
import unittest
import math

from builtins import zip
from mock import Mock, call, patch

from flightdatautilities import units as ut
from flightdatautilities.aircrafttables.interfaces import VelocitySpeed
from flightdatautilities.array_operations import load_compressed
from flightdatautilities.geometry import midpoint

from hdfaccess.parameter import MappedArray

from analysis_engine.node import (
    A, M, P, S, KPV, KTI, aeroplane, App, ApproachItem,
    KeyPointValue, KeyTimeInstance, load, helicopter,
    MultistateDerivedParameterNode, Section, SectionNode
)

from analysis_engine.library import (
    align,
    any_of,
    max_value,
    max_abs_value,
    min_value,
    median_value,
    np_ma_ones_like,
    runs_of_ones
)

from analysis_engine.multistate_parameters import StableApproach

from analysis_engine.key_point_values import (
    AOADuringGoAroundMax,
    AOAWithFlapDuringClimbMax,
    AOAWithFlapDuringDescentMax,
    AOAWithFlapMax,
    AOADifference5SecMax,
    AOAFlapsUpAPOffMax,
    AOAAbnormalOperationDuration,
    AOABelowStickShakerAOAMin,
    AOAStickShakerMinusAOAFlapsUpAPOffMin,
    TrimDownWhileControlColumnUpDuration,
    APDisengagedDuringCruiseDuration,
    APUOnDuringFlightDuration,
    APUFireWarningDuration,
    AccelerationLateralAtTouchdown,
    AccelerationLateralDuringLandingMax,
    AccelerationLateralDuringTakeoffMax,
    AccelerationLateralFor5SecMax,
    AccelerationLateralInTurnDuringTaxiInMax,
    AccelerationLateralInTurnDuringTaxiOutMax,
    AccelerationLateralMax,
    AccelerationLateralOffset,
    AccelerationLateralWhileAirborneMax,
    AccelerationLateralWhileTaxiingStraightMax,
    AccelerationLateralWhileTaxiingTurnMax,
    AccelerationLongitudinalWhileAirborneMax,
    AccelerationLongitudinalDuringLandingMin,
    AccelerationLongitudinalDuringTakeoffMax,
    AccelerationLongitudinalOffset,
    AccelerationNormal20FtTo5FtMax,
    AccelerationNormalAboveLimitWithFlapDownWhileAirborne,
    AccelerationNormalAboveWeightLimitAtTouchdown,
    AccelerationNormalAboveWeightLowLimitAtTouchdown,
    AccelerationNormalAboveWeightHighLimitAtTouchdown,
    AccelerationNormalAtLiftoff,
    AccelerationNormalAtTouchdown,
    AccelerationNormalMinusLoadFactorThresholdAtTouchdown,
    AccelerationNormalMax,
    AccelerationNormalOffset,
    AccelerationNormalLiftoffTo35FtMax,
    AccelerationNormalWhileAirborneMax,
    AccelerationNormalWhileAirborneMin,
    AccelerationNormalWithFlapDownWhileAirborneMax,
    AccelerationNormalWithFlapDownWhileAirborneMin,
    AccelerationNormalWithFlapUpWhileAirborneMax,
    AccelerationNormalWithFlapUpWhileAirborneMin,
    AileronPreflightCheck,
    AircraftEnergyWhenDescending,
    Airspeed10000To5000FtMax,
    Airspeed10000To8000FtMax,
    Airspeed1000To500FtMax,
    Airspeed1000To500FtMin,
    Airspeed1000To5000FtMax,
    Airspeed1000To8000FtMax,
    Airspeed3000FtToTopOfClimbMax,
    Airspeed3000FtToTopOfClimbMin,
    Airspeed3000To1000FtMax,
    Airspeed35To1000FtMax,
    Airspeed35To1000FtMin,
    Airspeed5000To3000FtMax,
    Airspeed500To20FtMax,
    Airspeed500To50FtMedian,
    Airspeed500To50FtMedianMinusAirspeedSelected,
    Airspeed500To20FtMin,
    Airspeed5000To8000FtMax,
    Airspeed5000To10000FtMax,
    Airspeed8000To10000FtMax,
    Airspeed8000To5000FtMax,
    AirspeedAboveStickShakerSpeedMin,
    AirspeedAtAPUpperModesEngaged,
    AirspeedAt35FtDuringTakeoff,
    AirspeedAt8000FtDescending,
    AirspeedAtFlapExtensionWithGearDownSelected,
    AirspeedAtGearDownSelection,
    AirspeedAtGearUpSelection,
    AirspeedAtLiftoff,
    AirspeedAtThrustReversersSelection,
    AirspeedAtTouchdown,
    AirspeedBelow10000FtDuringDescentMax,
    AirspeedDifference5SecMax,
    AirspeedDuringCruiseMax,
    AirspeedDuringCruiseMin,
    AirspeedDuringLevelFlightMax,
    AirspeedDuringRejectedTakeoffMax,
    AirspeedGustsDuringFinalApproach,
    AirspeedMax,
    AirspeedMinsToTouchdown,
    AirspeedMinusAirspeedSelectedFor3Sec500To20FtMax,
    AirspeedMinusAirspeedSelectedFor3Sec500To20FtMin,
    AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMax,
    AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMin,
    AirspeedMinusFlapManoeuvreSpeedWithFlapDuringDescentMin,
    AirspeedMinusMinimumAirspeedAbove10000FtMin,
    AirspeedMinusMinimumAirspeed35To10000FtMin,
    AirspeedMinusMinimumAirspeed10000To50FtMin,
    AirspeedMinusMinimumAirspeedDuringGoAroundMin,
    AirspeedMinusV235To1000FtMax,
    AirspeedMinusV235To1000FtMin,
    AirspeedMinusV235ToClimbAccelerationStartMax,
    AirspeedMinusV235ToClimbAccelerationStartMin,
    AirspeedMinusV2At35FtDuringTakeoff,
    AirspeedMinusV2AtLiftoff,
    AirspeedMinusV2For3Sec35To1000FtMax,
    AirspeedMinusV2For3Sec35To1000FtMin,
    AirspeedMinusV2For3Sec35ToClimbAccelerationStartMax,
    AirspeedMinusV2For3Sec35ToClimbAccelerationStartMin,
    AirspeedMinusVMOMax,
    AirspeedMinusVLSAtTouchdown,
    AirspeedMinusVLS1000To500FtMax,
    AirspeedMinusVLS1000To500FtMin,
    AirspeedMinusVLS500To20FtMax,
    AirspeedMinusVLS500To20FtMin,
    AirspeedMinusVLSFor3Sec1000To500FtMax,
    AirspeedMinusVLSFor3Sec1000To500FtMin,
    AirspeedMinusVLSFor3Sec500To20FtMax,
    AirspeedMinusVLSFor3Sec500To20FtMin,
    AirspeedRelative1000To500FtMax,
    AirspeedRelative1000To500FtMin,
    AirspeedRelative20FtToTouchdownMax,
    AirspeedRelative20FtToTouchdownMin,
    AirspeedRelative500To20FtMax,
    AirspeedRelative500To20FtMin,
    AirspeedRelativeAtTouchdown,
    AirspeedRelativeFor3Sec1000To500FtMax,
    AirspeedRelativeFor3Sec1000To500FtMin,
    AirspeedRelativeFor3Sec20FtToTouchdownMax,
    AirspeedRelativeFor3Sec20FtToTouchdownMin,
    AirspeedRelativeFor3Sec500To20FtMax,
    AirspeedRelativeFor3Sec500To20FtMin,
    AirspeedRelativeWithConfigurationDuringDescentMin,
    AirspeedSelectedAtLiftoff,
    AirspeedSelectedAtTakeoffAccelerationStart,
    AirspeedSelectedFMCMinusFlapManoeuvreSpeed1000to5000FtMin,
    AirspeedTopOfDescentTo10000FtMax,
    AirspeedTopOfDescentTo4000FtMax,
    AirspeedTopOfDescentTo4000FtMin,
    AirspeedTrueAtTouchdown,
    AirspeedVacatingRunway,
    AirspeedWhileGearExtendingMax,
    AirspeedWhileGearRetractingMax,
    AirspeedWithConfigurationMax,
    AirspeedWithConfiguration1FExcludingTransitionMax,
    AirspeedWithFlapAndSlatExtendedMax,
    AirspeedWithFlapIncludingTransition20AndSlatFullyExtendedMax,
    AirspeedWithFlapDuringClimbMax,
    AirspeedWithFlapDuringClimbMin,
    AirspeedWithFlapDuringDescentMax,
    AirspeedWithFlapDuringDescentMin,
    AirspeedWithFlapMax,
    AirspeedWithFlapMin,
    AirspeedWithGearDownMax,
    AirspeedWithSpeedbrakeDeployedMax,
    AirspeedWithThrustReversersDeployedMin,
    AirspeedAboveFL200Max,
    AirspeedAboveFL200Min,
    AlphaFloorDuration,
    AlternateLawDuration,
    AltitudeAtAPDisengagedSelection,
    AltitudeAtAPEngagedSelection,
    AltitudeAtATDisengagedSelection,
    AltitudeAtATEngagedSelection,
    #AltitudeAtCabinPressureLowWarningDuration,
    AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft,
    AltitudeAtClimbThrustDerateSelection,
    AltitudeAtFirstAPEngagedAfterLiftoff,
    AltitudeAtFirstFlapChangeAfterLiftoff,
    AltitudeAtFirstFlapExtensionAfterLiftoff,
    AltitudeAtFirstFlapRetraction,
    AltitudeAtFirstFlapRetractionDuringGoAround,
    AltitudeAtFlapExtension,
    AltitudeAtFlapExtensionWithGearDownSelected,
    AltitudeAtLastGearDownSelection,
    AltitudeAtGearDownSelectionWithFlapDown,
    AltitudeAtGearDownSelectionWithFlapUp,
    AltitudeAtFirstGearUpSelection,
    AltitudeAtGearUpSelectionDuringGoAround,
    AltitudeAtLastAPDisengagedDuringApproach,
    AltitudeAtLastFlapChangeBeforeTouchdown,
    AltitudeAtLastFlapSelectionBeforeTouchdown,
    AltitudeAtLastFlapRetraction,
    AltitudeAtMachMax,
    AltitudeDuringGoAroundMin,
    AltitudeFirstStableDuringApproachBeforeGoAround,
    AltitudeFirstStableDuringLastApproach,
    AltitudeLastUnstableDuringApproachBeforeGoAround,
    AltitudeLastUnstableDuringLastApproach,
    AltitudeLastUnstableDuringLastApproachExcludingEngThrust,
    AltitudeMax,
    AltitudeOvershootAtSuspectedLevelBust,
    AltitudeAALCleanConfigurationMin,
    AltitudeWithFlapMax,
    AltitudeSTDWithGearDownMax,
    AltitudeSTDMax,
    AltitudeWithGearDownMax,
    AltitudeInCruiseAverage,
    ATEngagedAPDisengagedOutsideClimbDuration,
    ATDisengagedAPEngagedDuration,
    AutobrakeRejectedTakeoffNotSetDuringTakeoff,
    BaroDifferenceDuration,
    BaroCorrectionMinus1013Above20000FtDuringLevelFlightMax,
    BrakePressureInTakeoffRollMax,
    BrakeTempAfterTouchdownDelta,
    BrakeTempDuringTaxiInMax,
    BrakeTempBeforeTakeoffMax,
    HeightAtDistancesFromThreshold,
    HeightAtOffsetILSTurn,
    HeightAtRunwayChange,
    CargoSmokeOrFireWarningDuration,
    ControlColumnForceMax,
    ControlColumnStiffness,
    ControlWheelForceMax,
    ControlColumnDualInputOppositeDirectionForceMax,
    DecelerationAfterTouchdownAvg,
    DecelerationFromTouchdownToStopOnRunway,
    DelayedBrakingAfterTouchdown,
    DHSelectedAt1500FtLVO,
    DirectLawDuration,
    DistanceFromRunwayCentrelineAtTouchdown,
    DistanceFromRunwayCentrelineFromTouchdownTo60KtMax,
    DistanceTravelledDuringTurnback,
    DistanceTravelledFollowingDiversion,
    DualInputAbove200FtDuration,
    DualInputBelow200FtDuration,
    DualInputByCaptDuration,
    DualInputByCaptMax,
    DualInputByFODuration,
    DualInputByFOMax,
    DualInputWarningDuration,
    ElevatorDuringLandingMin,
    ElevatorPreflightCheck,
    EngBleedValvesAtLiftoff,
    EngChipDetectorWarningDuration,
    EngEngStartToN1At60PercentDuration,
    EngEPR500To50FtMax,
    EngEPR500To50FtMin,
    EngEPRAtTOGADuringTakeoffMax,
    EngEPRDuringApproachMax,
    EngEPRDuringApproachMin,
    EngEPRDuringGoAround5MinRatingMax,
    EngEPRDuringMaximumContinuousPowerMax,
    EngEPRDuringTakeoff5MinRatingMax,
    EngEPRDuringTaxiMax,
    EngEPRDuringTaxiInMax,
    EngEPRDuringTaxiOutMax,
    EngEPRExceedEPRRedlineDuration,
    EngEPRFor5Sec1000To500FtMin,
    EngEPRFor5Sec500To50FtMin,
    EngEPRFor5SecDuringGoAround5MinRatingMax,
    EngEPRFor5SecDuringMaximumContinuousPowerMax,
    EngEPRFor5SecDuringTakeoff5MinRatingMax,
    EngEPRLowDuringTakeoff,
    EngFireWarningDuration,
    EngGasTempAboveNormalMaxLimitDuringTakeoff5MinRatingDuration,
    EngGasTempAboveNormalMaxLimitDuringMaximumContinuousPowerDuration,
    EngGasTempDuringEngStartForXSecMax,
    EngGasTempDuringEngStartMax,
    EngGasTempDuringFlightMin,
    EngGasTempDuringGoAround5MinRatingMax,
    EngGasTempDuringMaximumContinuousPowerForXMinMax,
    EngGasTempDuringMaximumContinuousPowerMax,
    EngGasTempMaxDuringTakeoffMaxMaintained,
    EngGasTempDuringTakeoff5MinRatingMax,
    EngGasTempExceededEngGasTempRedlineDuration,
    EngGasTempFor5SecDuringGoAround5MinRatingMax,
    EngGasTempFor5SecDuringMaximumContinuousPowerMax,
    EngGasTempFor5SecDuringTakeoff5MinRatingMax,
    EngGasTempOverThresholdDuration,
    EngN1500To50FtMax,
    EngN1500To50FtMin,
    EngN154to72PercentWithThrustReversersDeployedDurationMax,
    EngN1AtTOGADuringTakeoff,
    EngN1Below60PercentAfterTouchdownDuration,
    EngAPRDuration,
    EngN1CyclesDuringFinalApproach,
    EngN1DuringApproachMax,
    EngN1DuringGoAround5MinRatingMax,
    EngN1DuringMaximumContinuousPowerMax,
    EngN1DuringTakeoff5MinRatingMax,
    EngN1DuringTaxiMax,
    EngN1DuringTaxiInMax,
    EngN1DuringTaxiOutMax,
    EngN1ExceededN1RedlineDuration,
    EngN1For5Sec1000To500FtMin,
    EngN1For5Sec500To50FtMin,
    EngN1For5SecDuringGoAround5MinRatingMax,
    EngN1For5SecDuringMaximumContinuousPowerMax,
    EngN1For5SecDuringTakeoff5MinRatingMax,
    EngN1OverThresholdDuration,
    EngN1TakeoffDerate,
    EngN1WithThrustReversersDeployedMax,
    EngN1WithThrustReversersInTransitMax,
    EngN2CyclesDuringFinalApproach,
    EngN2DuringGoAround5MinRatingMax,
    EngN2DuringMaximumContinuousPowerMax,
    EngN2DuringTakeoffForXSecMax,
    EngN2DuringMaximumContinuousPowerForXSecMax,
    EngN2DuringTakeoff5MinRatingMax,
    EngN2DuringTaxiMax,
    EngN2ExceededN2RedlineDuration,
    EngN2For5SecDuringGoAround5MinRatingMax,
    EngN2For5SecDuringMaximumContinuousPowerMax,
    EngN2For5SecDuringTakeoff5MinRatingMax,
    EngN2OverThresholdDuration,
    EngN3DuringGoAround5MinRatingMax,
    EngN3DuringMaximumContinuousPowerMax,
    EngN3DuringTakeoff5MinRatingMax,
    EngN3DuringTaxiMax,
    EngN3ExceededN3RedlineDuration,
    EngN3For5SecDuringGoAround5MinRatingMax,
    EngN3For5SecDuringMaximumContinuousPowerMax,
    EngN3For5SecDuringTakeoff5MinRatingMax,
    EngNp82To90PercentDurationMax,
    EngNpDuringClimbMin,
    EngNpDuringGoAround5MinRatingMax,
    EngNpDuringMaximumContinuousPowerMax,
    EngNpDuringTakeoff5MinRatingMax,
    EngNpFor5SecDuringGoAround5MinRatingMax,
    EngNpForXSecDuringMaximumContinuousPowerMax,
    EngNpFor5SecDuringTakeoff5MinRatingMax,
    EngNpDuringTaxiMax,
    EngNpOverThresholdDuration,
    EngNpMaxDuringTakeoff,
    EngVibNpMax,
    EngOilPressFor60SecDuringCruiseMax,
    EngOilPressMax,
    EngOilPressMin,
    EngOilPressWarningDuration,
    EngOilPressLowRedlineExceededDuration,
    EngOilQtyDuringTaxiInMax,
    EngOilQtyDuringTaxiOutMax,
    EngOilQtyMax,
    EngOilQtyMin,
    EngOilTempForXMinMax,
    EngOilTempMax,
    EngRunningDuration,
    EngShutdownDuringFlightDuration,
    EngStartTimeMax,
    EngTPRAtTOGADuringTakeoffMin,
    EngTPRDuringGoAround5MinRatingMax,
    EngTPRDuringMaximumContinuousPowerMax,
    EngTPRDuringTakeoff5MinRatingMax,
    EngTPRFor5SecDuringGoAround5MinRatingMax,
    EngTPRFor5SecDuringMaximumContinuousPowerMax,
    EngTPRFor5SecDuringTakeoff5MinRatingMax,
    EngTakeoffFlexTemp,
    EngThrustTakeoffDerate,
    EngTorque500To50FtMax,
    EngTorque500To50FtMin,
    EngTorqueDuringGoAround5MinRatingMax,
    EngTorqueDuringMaximumContinuousPowerMax,
    EngTorqueDuringTakeoff5MinRatingMax,
    EngTorqueDuringTaxiMax,
    EngTorqueFor5SecDuringGoAround5MinRatingMax,
    EngTorqueFor5SecDuringMaximumContinuousPowerMax,
    EngTorqueFor5SecDuringTakeoff5MinRatingMax,
    EngTorqueOverThresholdDuration,
    EngTorqueLimitExceedanceWithOneEngineInoperativeDuration,
    EngTorqueMaxDuringMaximumContinuousPower,
    EngTorqueMaxDuringTakeoff,
    EngTorqueWhileDescendingMax,
    EngTorque7FtToTouchdownMax,
    EngTorqueWithin1SecOfTouchdownMax,
    EngVibAMax,
    EngVibBMax,
    EngVibBroadbandMax,
    EngVibCMax,
    EngVibN1Max,
    EngVibN2Max,
    EngVibN3Max,
    FlapAt1000Ft,
    FlapAt500Ft,
    FlapAtGearDownSelection,
    FlapAtGearUpSelectionDuringGoAround,
    FlapLeverAtGearDownSelection,
    FlapLeverAtGearUpSelectionDuringGoAround,
    FlapAtFirstMovementAfterEngineStart,
    FlapAtLiftoff,
    FlapAtTouchdown,
    FlapOrConfigurationMaxOrMin,
    FlapWithGearUpMax,
    FlapWithSpeedbrakeDeployedMax,
    FlapSynchroAsymmetryMax,
    FlapBypassValveDuration,
    FlareDistance20FtToTouchdown,
    FlareDuration20FtToTouchdown,
    FlightControlPreflightCheck,
    FuelCrossFeedValveStateAtLiftoff,
    ApproachFlightPathAngle1500To1000FtMax,
    ApproachFlightPathAngle1500To1000FtMin,
    ApproachFlightPathAngle1000To500FtMax,
    ApproachFlightPathAngle1000To500FtMin,
    ApproachFlightPathAngle500To200FtMax,
    ApproachFlightPathAngle500To200FtMin,
    FuelJettisonDuration,
    FuelQtyAtLiftoff,
    FuelQtyAtTouchdown,
    FuelQtyLowWarningDuration,
    FuelQtyWingDifferenceMax,
    FuelQtyWingDifference787Max,
    GearboxChipDetectorWarningDuration,
    GearDownToLandingFlapConfigurationDuration,
    GreatCircleDistance,
    GrossWeightAtLiftoff,
    GrossWeightAtTouchdown,
    GrossWeightConditionalAtTouchdown,
    GrossWeightDelta60SecondsInFlightMax,
    GroundspeedAtLiftoff,
    GroundspeedAtTOGA,
    GroundspeedAtTouchdown,
    GroundspeedDuringRejectedTakeoffMax,
    GroundspeedFlapChangeDuringTakeoffMax,
    GroundspeedInStraightLineDuringTaxiInMax,
    GroundspeedInStraightLineDuringTaxiOutMax,
    GroundspeedInTurnDuringTaxiInMax,
    GroundspeedInTurnDuringTaxiOutMax,
    GroundspeedWithGearOnGroundMax,
    GroundspeedSpeedbrakeDuringTakeoffMax,
    GroundspeedSpeedbrakeHandleDuringTakeoffMax,
    GroundspeedStabilizerOutOfTrimDuringTakeoffMax,
    GroundspeedVacatingRunway,
    GroundspeedWhileTaxiingStraightMax,
    GroundspeedWhileTaxiingTurnMax,
    GroundspeedWithThrustReversersDeployedAnyPowerMin,
    GroundspeedWithThrustReversersDeployedMin,
    HeadingAtLowestAltitudeDuringApproach,
    HeadingChange,
    HeadingDeviationFromRunwayAt50FtDuringLanding,
    HeadingDeviationFromRunwayAtTOGADuringTakeoff,
    HeadingDeviationFromRunwayDuringLandingRoll,
    HeadingDuringLanding,
    HeadingDuringTakeoff,
    HeadingRateWhileAirborneMax,
    HeadingTrueDuringLanding,
    HeadingTrueDuringTakeoff,
    HeadingVacatingRunway,
    HeadingVariation300To50Ft,
    HeadingVariation500To50Ft,
    HeadingVariation800To50Ft,
    HeadingVariationAbove80KtsAirspeedDuringTakeoff,
    HeadingVariationAbove100KtsAirspeedDuringLanding,
    HeadingVariationTouchdownPlus4SecTo60KtsAirspeed,
    HeightLoss1000To2000Ft,
    HeightLoss35To1000Ft,
    HeightLossLiftoffTo35Ft,
    HeightMinsToTouchdown,
    HeightSelectedOnApproachMin,
    IANFinalApproachCourseDeviationMax,
    IANGlidepathDeviationMax,
    ILSFrequencyDuringApproach,
    ILSGlideslope10SecBeforeEstablished,
    ILSGlideslopeDeviation1000To500FtMax,
    ILSGlideslopeDeviation1500To1000FtMax,
    ILSGlideslopeDeviation500To200FtMax,
    ILSGlideslopeNotEstablishedHighestAltitude1000To200Ft,
    ILSGlideslopeNotEstablishedHighestAltitude500To200Ft,
    ILSLocalizerDeviation1000To500FtMax,
    ILSLocalizerDeviation1500To1000FtMax,
    ILSLocalizerDeviation500To200FtMax,
    ILSLocalizerDeviationAtTouchdown,
    ILSLocalizerNotEstablishedHighestAltitude1000To200Ft,
    ILSLocalizerNotEstablishedHighestAltitude500To200Ft,
    IsolationValveOpenAtLiftoff,
    KineticEnergyAtRunwayTurnoff,
    LandingConfigurationGearWarningDuration,
    LandingConfigurationSpeedbrakeCautionDuration,
    LastFlapChangeToTakeoffRollEndDuration,
    LastUnstableStateDuringApproachBeforeGoAround,
    LastUnstableStateDuringLastApproach,
    LatitudeAtLiftoff,
    LatitudeAtLowestAltitudeDuringApproach,
    LatitudeAtTakeoffAccelerationStart,
    LatitudeAtTouchdown,
    LatitudeOffBlocks,
    LatitudeSmoothedAtLiftoff,
    LatitudeSmoothedAtTouchdown,
    LoadFactorThresholdAtTouchdown,
    LongitudeAtLiftoff,
    LongitudeAtLowestAltitudeDuringApproach,
    LongitudeAtTakeoffAccelerationStart,
    LongitudeAtTouchdown,
    LongitudeOffBlocks,
    LongitudeSmoothedAtLiftoff,
    LongitudeSmoothedAtTouchdown,
    MachDuringCruiseAvg,
    MachMax,
    MachMinusMMOMax,
    MachWhileGearExtendingMax,
    MachWhileGearRetractingMax,
    MachWithFlapMax,
    MachWithGearDownMax,
    MachAboveFL200Max,
    MachAboveFL200Min,
    MagneticVariationAtLandingTurnOffRunway,
    MagneticVariationAtTakeoffTurnOntoRunway,
    MainFuelQuantityOffBlockWithFuelInCenterTank,
    MainGearOnGroundToNoseGearOnGroundDuration,
    MasterCautionDuringTakeoffDuration,
    MasterWarningDuration,
    MasterWarningDuringTakeoffDuration,
    NumberOfAPChannelsEngagedAtTouchdown,
    OverspeedDuration,
    StallFaultCautionDuration,
    CruiseSpeedLowDuration,
    DegradedPerformanceCautionDuration,
    AirspeedIncreaseAlertDuration,
    AirspeedBelowMinimumAirspeedFlapCleanMin,
    PackValvesOpenAtLiftoff,
    PercentApproachStable,
    Pitch1000To500FtMax,
    Pitch1000To500FtMin,
    Pitch20FtToTouchdownMax,
    Pitch20FtToTouchdownMin,
    Pitch35ToClimbAccelerationStartMax,
    Pitch35ToClimbAccelerationStartMin,
    Pitch35To400FtMax,
    Pitch35To400FtMin,
    Pitch400To1000FtMax,
    Pitch400To1000FtMin,
    Pitch500To20FtMin,
    Pitch500To7FtMax,
    Pitch500To7FtMin,
    Pitch500To50FtMax,
    Pitch50FtToTouchdownMax,
    Pitch7FtToTouchdownMin,
    Pitch7FtToTouchdownMax,
    PitchAbove1000FtMin,
    PitchAbove1000FtMax,
    PitchAfterFlapRetractionMax,
    PitchAt35FtDuringClimb,
    PitchAtLiftoff,
    PitchAtTouchdown,
    PitchCyclesDuringFinalApproach,
    PitchDisconnectDuration,
    PitchDuringGoAroundMax,
    Pitch500To50FtMin,
    PitchWhileAirborneMax,
    PitchWhileAirborneMin,
    PitchTouchdownTo60KtsAirspeedMax,
    PitchAboveFL200Min,
    PitchAboveFL200Max,
    PitchRateTouchdownTo60KtsAirspeedMax,
    PitchRate20FtToTouchdownMax,
    PitchRate20FtToTouchdownMin,
    PitchRate2DegPitchTo35FtMax,
    PitchRate2DegPitchTo35FtMin,
    PitchRate35To1000FtMax,
    PitchRate35ToClimbAccelerationStartMax,
    PitchRateWhileAirborneMax,
    PitchTakeoffMax,
    QNHDifferenceDuringApproach,
    RateOfClimb35To1000FtMin,
    RateOfClimb35ToClimbAccelerationStartMin,
    RateOfClimbBelow10000FtMax,
    RateOfClimbDuringGoAroundMax,
    RateOfClimbMax,
    RateOfClimbAtHeightBeforeLevelFlight,
    RateOfDescent10000To5000FtMax,
    RateOfDescent1000To500FtMax,
    RateOfDescent2000To1000FtMax,
    RateOfDescent3000To2000FtMax,
    RateOfDescent5000To3000FtMax,
    RateOfDescent500To50FtMax,
    RateOfDescent50FtToTouchdownMax,
    RateOfDescentAtTouchdown,
    RateOfDescentBelow80KtsMax,
    RateOfDescentBelow10000FtMax,
    RateOfDescentMax,
    RateOfDescentAbove3000FtMax,
    RateOfDescentTopOfDescentTo10000FtMax,
    RateOfDescentAtHeightBeforeLevelFlight,
    Roll1000To300FtMax,
    Roll20FtToTouchdownMax,
    Roll20To400FtMax,
    Roll300To20FtMax,
    Roll400To1000FtMax,
    RollAbove1000FtMax,
    RollAtLowAltitude,
    RollCyclesExceeding5DegDuringFinalApproach,
    RollCyclesExceeding15DegDuringFinalApproach,
    RollCyclesExceeding5DegDuringInitialClimb,
    RollCyclesExceeding15DegDuringInitialClimb,
    RollCyclesNotDuringFinalApproach,
    RollLeftAbove8000FtAltitudeDensityAbove60Kts,
    RollLeftAbove6000FtAltitudeDensityBelow60Kts,
    RollLeftBelow8000FtAltitudeDensityAbove60Kts,
    RollLeftBelow6000FtAltitudeDensityBelow60Kts,
    RollLiftoffTo20FtMax,
    RollAboveFL200Max,
    RollRateMaxAboveLimitAtTouchdown,
    RudderCyclesAbove50Ft,
    RudderDuringTakeoffMax,
    RudderPedalForceMax,
    RudderPreflightCheck,
    RudderReversalAbove50Ft,
    SATMax,
    SingleEngineDuringTaxiInDuration,
    SingleEngineDuringTaxiOutDuration,
    SmokeWarningDuration,
    SpeedbrakeDeployed1000To20FtDuration,
    AltitudeWithSpeedbrakeDeployedDuringFinalApproachMin,
    SpeedbrakeDeployedDuringGoAroundDuration,
    SpeedbrakeDeployedWithFlapDuration,
    SpeedbrakeDeployedWithPowerOnDuration,
    SpoilersDeployedDurationDuringLanding,
    StallWarningDuration,
    StickPusherActivatedDuration,
    StickShakerActivatedDuration,
    TailwindAtAltitudeDuringDescent,
    TAWSAlertDuration,
    TAWSCautionObstacleDuration,
    TAWSCautionTerrainDuration,
    TAWSDontSinkWarningDuration,
    TAWSFailureDuration,
    TAWSGeneralWarningDuration,
    TAWSGlideslopeWarning1000To500FtDuration,
    TAWSGlideslopeWarning1500To1000FtDuration,
    TAWSGlideslopeWarning500To200FtDuration,
    TAWSObstacleWarningDuration,
    TAWSPredictiveWindshearDuration,
    TAWSPullUpWarningDuration,
    TAWSSinkRateWarningDuration,
    TAWSUnspecifiedDuration,
    TAWSTerrainAheadDuration,
    TAWSTerrainAheadPullUpDuration,
    TAWSTerrainCautionDuration,
    TAWSTerrainPullUpWarningDuration,
    TAWSTerrainWarningDuration,
    TAWSTerrainClearanceFloorAlertDuration,
    TAWSTooLowFlapWarningDuration,
    TAWSTooLowGearWarningDuration,
    TAWSTooLowTerrainWarningDuration,
    TAWSWarningDuration,
    TAWSWindshearCautionBelow1500FtDuration,
    TAWSWindshearSirenBelow1500FtDuration,
    TAWSWindshearWarningBelow1500FtDuration,
    TCASFailureRatio,
    TCASRAAPDisengaged,
    TCASRAToAPDisengagedDelay,
    TCASRAAcceleration,
    TCASRAAltitudeAAL,
    TCASRAAltitudeSTD,
    TCASRAChangeOfVerticalSpeed,
    TCASRADirection,
    TCASRAErroneousAcceleration,
    TCASRAHeading,
    TCASRAReactionDelay,
    TCASRASubsequentAcceleration,
    TCASRASubsequentReactionDelay,
    TCASRAWarningDuration,
    TCASRAWarningAboveFL100Duration,
    TCASRAWarningBelowFL100InClimbDuration,
    TCASRAWarningBelowFL100InDescentDuration,
    TCASTAAcceleration,
    TCASTAAltitudeAAL,
    TCASTAAltitudeSTD,
    TCASTAHeading,
    TCASTAWarningDuration,
    TOGASelectedDuringFlightDuration,
    TOGASelectedDuringGoAroundDuration,
    TailClearanceDuringApproachMin,
    TailClearanceDuringLandingMin,
    TailClearanceDuringTakeoffMin,
    TailwindDuringTakeoffMax,
    Tailwind100FtToTouchdownMax,
    TailwindLiftoffTo100FtMax,
    TakeoffConfigurationFlapWarningDuration,
    TakeoffConfigurationParkingBrakeWarningDuration,
    TakeoffConfigurationSpoilerWarningDuration,
    TakeoffConfigurationStabilizerWarningDuration,
    TakeoffConfigurationWarningDuration,
    TakeoffRatingDuration,
    TaxiInDuration,
    TaxiOutDuration,
    TerrainClearanceAbove3000FtMin,
    ThrottleCyclesDuringFinalApproach,
    ThrottleLeverAtLiftoff,
    ThrottleLeverVariationAbove80KtsToTakeoff,
    ThrottleReductionToTouchdownDuration,
    ThrustAsymmetryDuringApproachDuration,
    ThrustAsymmetryDuringApproachMax,
    ThrustAsymmetryDuringFlightMax,
    ThrustAsymmetryDuringGoAroundMax,
    ThrustAsymmetryDuringTakeoffMax,
    ThrustAsymmetryWithThrustReversersDeployedDuration,
    ThrustAsymmetryWithThrustReversersDeployedMax,
    ThrustRatingCLB1Duration,
    ThrustReversersCancelToEngStopDuration,
    ThrustReversersDeployedDuration,
    ThrustReversersDeployedDuringFlightDuration,
    TorqueAsymmetryWhileAirborneMax,
    TouchdownTo60KtsDuration,
    TouchdownToElevatorDownDuration,
    TouchdownToPitch2DegreesAbovePitchAt60KtsDuration,
    TouchdownToSpoilersDeployedDuration,
    TouchdownToThrustReversersDeployedDuration,
    TurbulenceDuringApproachMax,
    TurbulenceDuringCruiseMax,
    TurbulenceDuringFlightMax,
    TwoDegPitchTo35FtDuration,
    V2AtLiftoff,
    V2LookupAtLiftoff,
    WindAcrossLandingRunwayAt50Ft,
    WindDirectionAtAltitudeDuringDescent,
    WindSpeedAtAltitudeDuringDescent,
    DriftAtTouchdown,
    TransmitInactivityDuration
)

from analysis_engine.key_time_instances import (
    AltitudeWhenDescending,
    AltitudeBeforeLevelFlightWhenClimbing,
    AltitudeBeforeLevelFlightWhenDescending,
    EngStart,
    EngStop,
    DistanceFromThreshold
)

from analysis_engine.flight_phase import Fast, RejectedTakeoff
from analysis_engine.test_utils import buildsection, buildsections

debug = sys.gettrace() is not None

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


##############################################################################
# Superclasses


class NodeTest(object):

    def generate_attributes(self, manufacturer):
        if manufacturer == 'boeing':
            _am = A('Model', 'B737-333')
            _as = A('Series', 'B737-300')
            _af = A('Family', 'B737 Classic')
            _et = A('Engine Type', 'CFM56-3B1')
            _es = A('Engine Series', 'CFM56-3')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'airbus':
            _am = A('Model', 'A330-333')
            _as = A('Series', 'A330-300')
            _af = A('Family', 'A330')
            _et = A('Engine Type', 'Trent 772B-60')
            _es = A('Engine Series', 'Trent 772B')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'beechcraft':
            _am = A('Model', '1900D')
            _as = A('Series', '1900D')
            _af = A('Family', '1900')
            _et = A('Engine Type', 'PT6A-67D')
            _es = A('Engine Series', 'PT6A')
            return (_am, _as, _af, _et, _es)
        raise ValueError('Unexpected lookup for attributes.')

    def test_can_operate(self):
        if not hasattr(self, 'node_class'):
            return
        kwargs = getattr(self, 'can_operate_kwargs', {})
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations(**kwargs)),
                self.operational_combination_length,
            )
        else:
            combinations = list(map(set, self.node_class.get_operational_combinations(**kwargs)))
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)

    def get_params_from_hdf(self, hdf_path, param_names, _slice=None,
                            phase_name='Phase'):
        import shutil
        import tempfile
        from hdfaccess.file import hdf_file
        from analysis_engine.node import derived_param_from_hdf

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(temp_file.name) as hdf:
                for param_name in param_names:
                    p = hdf.get(param_name)
                    if p is not None:
                        p = derived_param_from_hdf(p)
                    params.append(p)

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


class CreateKPVsWhereTest(NodeTest):
    '''
    Basic test for KPVs created with `create_kpvs_where()` method.

    The rationale for this class is to be able to use very simple test case
    boilerplate for the "multi state parameter duration in given flight phase"
    scenario.

    This test checks basic mechanics of specific type of KPV: duration of a
    given state in multistate parameter.

    The test supports multiple parameters and optionally a phase name
    within which the duration is measured.

    What is tested this class:
        * kpv.can_operate() results
        * parameter and KPV names
        * state names
        * basic logic to measure the time of a state duration within a phase
          (slice)

    What isn't tested:
        * potential edge cases of specific parameters
    '''
    def basic_setup(self):
        '''
        Setup for test_derive_basic.

        In the most basic use case the test which derives from this class
        should declare the attributes used to build the test case and then call
        self.basic_setup().

        You need to declare:

        self.node_class::
            class of the KPV node to be used to derive

        self.param_name::
            name of the parameter to be passed to the KPVs `derive()` method

        self.phase_name::
            name of the flight phase to be passed to the `derive()` or None if
            the KPV does not use phases

        self.values_mapping::
            "state to state name" mapping for multistate parameter

        Optionally:

        self.additional_params::
            list of additional parameters to be passed to the `derive()` after
            the main parameter. If unset, only one parameter will be used.


        The method performs the following operations:

            1. Builds the main parameter using self.param_name,
               self.values_array and self.values_mapping

            2. Builds self.params list from the main parameter and
               self.additional_params, if given
            3. Optionally builds self.phases with self.phase_name if given
            4. Builds self.operational_combinations from self.params and
               self.phases
            5. Builds self.expected list of expected values using
               self.node_class and self.phases

        Any of the built attributes can be overridden in the derived class to
        alter the expected test results.
        '''
        if not hasattr(self, 'values_array'):
            self.values_array = np.ma.concatenate((np.zeros(3), np.ones(6), np.zeros(3)))

        if not hasattr(self, 'phase_slice'):
            self.phase_slice = slice(2, 7)

        if not hasattr(self, 'expected_index'):
            self.expected_index = 3

        if not hasattr(self, 'params'):
            self.params = [
                MultistateDerivedParameterNode(
                    self.param_name,
                    array=self.values_array,
                    values_mapping=self.values_mapping
                )
            ]

            if hasattr(self, 'additional_params'):
                self.params += self.additional_params

        if hasattr(self, 'phase_name') and self.phase_name:
            self.phases = buildsection(self.phase_name,
                                       self.phase_slice.start,
                                       self.phase_slice.stop)
        else:
            self.phases = []

        if not hasattr(self, 'operational_combinations'):
            combinations = [p.name for p in self.params]

            self.operational_combinations = [combinations]
            if self.phases:
                combinations.append(self.phases.name)

        if not hasattr(self, 'expected'):
            self.expected = []
            if self.phases:
                # TODO: remove after intervals have been implemented
                if hasattr(self, 'complex_where'):
                    slices = self.phases.get_slices()
                else:
                    slices = [p.slice for p in self.phases]
            else:
                slices = [slice(None)]

            for sl in slices:
                expected_value = np.count_nonzero(
                    self.values_array[sl])
                if expected_value:
                    self.expected.append(
                        KeyPointValue(
                            name=self.node_class().get_name(),
                            index=self.expected_index,
                            value=expected_value
                        )
                    )

    def test_can_operate(self):
        '''
        Test the operational combinations.
        '''
        # sets of sorted tuples of node combinations must match exactly
        kpv_operational_combinations = \
            self.node_class.get_operational_combinations()

        kpv_combinations = set(
            tuple(sorted(c)) for c in kpv_operational_combinations)

        expected_combinations = set(
            tuple(sorted(c)) for c in self.operational_combinations)

        self.assertSetEqual(kpv_combinations, expected_combinations)

    def test_derive_basic(self):
        if hasattr(self, 'node_class'):
            node = self.node_class()
            params = self.params
            if self.phases:
                params.append(self.phases)
            node.derive(*(params))
            self.assertEqual(node, self.expected)


class CreateKPVsAtKPVsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKPVsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_kpvs = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_kpvs.assert_called_once_with(mock1.array, mock2)


class CreateKPVsAtKTIsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        kwargs = {}
        if hasattr(self, 'interpolate'):
            kwargs = {'interpolate': self.interpolate}
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2, **kwargs)


class CreateKPVsWithinSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        mock1.frequency = 1.0
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpvs_within_slices = Mock()
        node.derive(mock1, mock2)
        if hasattr(self, 'second_param_method_calls'):
            mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
            node.create_kpvs_within_slices.assert_called_once_with(
                mock1.array, mock3.return_value, self.function)
        else:
            self.assertEqual(mock2.method_calls, [])
            node.create_kpvs_within_slices.assert_called_once_with(
                mock1.array, mock2, self.function)


class CreateKPVsWithinSlicesSecondWindowTest(CreateKPVsWithinSlicesTest):
    '''
    '''
    @patch('analysis_engine.key_point_values.second_window')
    def test_derive_mocked(self, second_window):
        # Not interested in testing functionallity of second window, this is
        # handled in library tests. Here we just want to check it was called
        # with the correct duration.
        second_window.side_effect = lambda *args, **kw: args[0]
        super(CreateKPVsWithinSlicesSecondWindowTest, self).test_derive_mocked()
        self.assertEqual(second_window.call_count, 1)
        # check correct duration used.
        self.assertEqual(second_window.call_args[0][2], self.duration, msg="Incorrect duration used.")


class CreateKPVFromSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpv_from_slices = Mock()
        node.derive(mock1, mock2)
        ####if hasattr(self, 'second_param_method_calls'):
        ####    mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
        ####    node.create_kpv_from_slices.assert_called_once_with(\
        ####        mock1.array, mock3.return_value, self.function)
        ####else:
        ####    self.assertEqual(mock2.method_calls, [])
        ####    node.create_kpv_from_slices.assert_called_once_with(\
        ####        mock1.array, mock2, self.function)


class ILSTest(NodeTest):
    '''
    '''

    def prepare__frequency__basic(self):
        # Let's give this a really hard time with alternate samples invalid and
        # the final signal only tuned just at the end of the data.
        ils_frequency = P(
            name='ILS Frequency',
            array=np.ma.concatenate((np.ones(6) * 108.5, np.ones(4) * 114.05)),
        )
        ils_frequency.array[0:10:2] = np.ma.masked
        ils_ests = buildsection('ILS Localizer Established', 2, 8)
        return ils_frequency, ils_ests

    def prepare__glideslope__basic(self):
        ils_glideslope = P(
            name='ILS Glideslope',
            array=1.0 - np.ma.cos(np.ma.arange(0, 6.3, 0.1)),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            # Altitude from 1875 to 325 ft in 63 steps.
            array=(75 - np.ma.arange(63)) * 25,
        )
        ils_ests = buildsection('ILS Glideslope Established', 2, 62)
        return ils_glideslope, ils_ests, alt_aal

    def prepare__glideslope__four_peaks(self):
        ils_glideslope = P(
            name='ILS Glideslope',
            array=-0.2 - np.ma.sin(np.ma.arange(0, 12.6, 0.1)),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            # Altitude from 1875 to 325 ft in 63 steps.
            array=(75 - np.ma.arange(63)) * 25,
        )
        ils_ests = buildsection('ILS Glideslope Established', 2, 55)
        return ils_glideslope, ils_ests, alt_aal

    def prepare__localizer__basic(self):
        ils_localizer = P(
            name='ILS Localizer',
            array=np.ma.arange(0, 12.6, 0.1),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.cos(np.ma.arange(0, 12.6, 0.1)) * -1000 + 1000,
        )
        ils_ests = buildsection('ILS Localizer Established', 30, 115)
        return ils_localizer, ils_ests, alt_aal


##############################################################################
# Test Classes


##############################################################################
# Acceleration


########################################
# Acceleration: Lateral


class TestAccelerationLateralMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralMax
        self.operational_combinations = [
            ('Acceleration Lateral',),
            ('Acceleration Lateral', 'Groundspeed'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralAtTouchdown
        self.operational_combinations = [('Acceleration Lateral', 'Touchdown')]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_lat = Mock()
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3, 'Touchdown'),
            KeyTimeInstance(1, 'Touchdown'),
        ])
        node = AccelerationLateralAtTouchdown()
        node.derive(acc_lat, touchdowns)
        bump.assert_has_calls([
            call(acc_lat, touchdowns[0].index),
            call(acc_lat, touchdowns[1].index),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Lateral At Touchdown'),
            KeyPointValue(1, 2.0, 'Acceleration Lateral At Touchdown'),
        ])


class TestAccelerationLateralDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLateralDuringTakeoffMax
        self.operational_combinations = [('Acceleration Lateral', 'Takeoff Roll')]
        self.function = max_abs_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLateralDuringLandingMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralDuringLandingMax
        self.operational_combinations = [('Acceleration Lateral', 'Landing Roll', 'FDR Landing Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLateralWhileAirborneMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralWhileAirborneMax
        self.operational_combinations = [('Acceleration Lateral', 'Airborne')]

    def test_derive(self):
        array = -0.1 + np.ma.sin(np.arange(0, 3.14*2, 0.04))
        accel_lat = P(name='Acceleration Lateral', array=array)
        airborne = buildsection('Airborne', 5, 150)
        node = self.node_class()
        node.derive(accel_lat, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 118)
        self.assertAlmostEqual(node[0].value, -1.1, places=1)


class TestAccelerationLateralWhileTaxiingStraightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralWhileTaxiingStraightMax
        self.operational_combinations = [('Acceleration Lateral Smoothed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralWhileTaxiingTurnMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralWhileTaxiingTurnMax
        self.operational_combinations = [('Acceleration Lateral Smoothed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralInTurnDuringTaxiInMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralInTurnDuringTaxiInMax
        self.operational_combinations = [('Acceleration Lateral Smoothed', 'Taxi In', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralInTurnDuringTaxiOutMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralInTurnDuringTaxiOutMax
        self.operational_combinations = [('Acceleration Lateral Smoothed', 'Taxi Out', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralOffset(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralOffset
        self.operational_combinations = [('Acceleration Lateral', 'Taxiing', 'Turning On Ground')]
        self.acc_lat_array=np.ma.array([0] * 17 + [0.02, 0.05, 0.02, 0, -0.017,] + [0] * 7 +
                                       [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 6 + [-0.02] +
                                           [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 10 +
                                           [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 10)*1.5
        self.taxiing = S(items=[Section('Taxiing', slice(10, 40), 10, 40),
                                Section('Taxiing', slice(60, 75), 60, 75)])
        self.turns = S(items=[Section('Turning On Ground', slice(17, 20), 17, 20),
                              Section('Turning On Ground', slice(29, 32), 29, 32),
                             Section('Turning On Ground', slice(71, 74), 71, 74)])
        #[slice(20, 29, None), slice(32, 40, None), slice(60, 71, None)]

    def test_derive_normal_scenario(self):
        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array)

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, 0.00755357142857, delta=0.00000000000001)

    def test_derive_negative_values(self):
        acc_lat = P(name='Acceleration Lateral', array=-self.acc_lat_array)

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, -0.00755357142857, delta=0.00000000000001)

    def test_derive_all_values_equal_0(self):
        acc_lat = P(name='Acceleration Lateral', array=np.zeros_like(self.acc_lat_array))

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, 0.0)

    def test_derive_values_in_given_slices_all_0(self):
        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array)
        acc_lat.array[10:40] = acc_lat.array[60:75] = 0.0

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, 0.0)

    def test_derive_values_in_one_slice_only(self):
        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array)
        acc_lat.array[10:40] = 0.0

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, 0.00610714285714, delta=0.00000000000001)

    def test_derive_not_enough_samples(self):
        turns = S(items=[Section('Turning On Ground', slice(17, 39), 17, 39),
                         Section('Turning On Ground', slice(61, 74), 61, 74)])

        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array)

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, turns)

        self.assertEqual(len(acc_lat_offset_kpv), 0)

    def test_derive_large_delta(self):
        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array+1.5)

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 1)
        self.assertAlmostEqual(acc_lat_offset_kpv[0].value, 1.50755357143, delta=0.00000000001)

    def test_derive_masked_data(self):
        acc_lat = P(name='Acceleration Lateral', array=self.acc_lat_array)
        acc_lat.array[15:75] = np.ma.masked

        acc_lat_offset_kpv = AccelerationLateralOffset()
        acc_lat_offset_kpv.derive(acc_lat, self.taxiing, self.turns)

        self.assertEqual(len(acc_lat_offset_kpv), 0)


class TestAccelerationLateralFor5SecMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationLateralFor5SecMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Lateral',)])

    def test_can_operate_787(self):
        self.assertFalse(
            self.node_class.can_operate(('Acceleration Lateral',),
                                        frame=A('Frame', value='787_frame'))
        )

    def test_derive(self):
        x = np.linspace(0, 10, 400)
        accel_lat = P(
            name='Acceleration Lateral',
            array=-x*np.sin(x),
            frequency=4
        )

        node = self.node_class()
        node.derive(accel_lat)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 308)
        self.assertAlmostEqual(node[0].value, -7.649, places=3)


########################################
# Acceleration: Longitudinal


class TestAccelerationLongitudinalOffset(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalOffset
        self.operational_combinations = [('Acceleration Longitudinal', 'Mobile', 'Fast')]
        self.acc_lon_array=np.ma.array([0] * 17 + [0.02, 0.05, 0.02, 0, -0.017,] + [0] * 7 +
                                       [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 6 + [-0.02] +
                                       [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 10 +
                                       [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 10)*1.5
        self.mobiles = S(items=[Section('Mobile', slice(10, 40), 10, 40),
                                Section('Mobile', slice(60, 75), 60, 75)])
        self.fasts = S(items=[Section('Fast', slice(17, 20), 17, 20),
                              Section('Fast', slice(29, 32), 29, 32),
                              Section('Fast', slice(71, 74), 71, 74)])
        #slice(10, 17), slice(20, 29), slice(32, 40), slice(60, 71), slice(74, 75)

    def test_derive_normal_scenario(self):
        acc_lon = P(name='Acceleration Longitudinal', array=self.acc_lon_array)

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, 0.005875, delta=0.00000000000001)

    def test_derive_negative_values(self):
        acc_lon = P(name='Acceleration Longitudinal', array=-self.acc_lon_array)

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, -0.005875, delta=0.00000000000001)

    def test_derive_all_values_equal_0(self):
        acc_lon = P(name='Acceleration Longitudinal', array=np.zeros_like(self.acc_lon_array))

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, 0.0)

    def test_derive_values_in_given_slices_all_0(self):
        acc_lon = P(name='Acceleration Longitudinal', array=-self.acc_lon_array)
        acc_lon.array[10:17] = acc_lon.array[20:29] = acc_lon.array[32:40] = \
            acc_lon.array[60:71] = acc_lon.array[74:75] = 0.0

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, 0.0)

    def test_derive_values_in_one_mobile_slice_only(self):
        acc_lon = P(name='Acceleration Longitudinal', array=self.acc_lon_array)
        acc_lon.array[60:71] = acc_lon.array[74:75] = 0.0

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, 0.001125)

    def test_derive_not_enough_samples(self):
        acc_lon = P(name='Acceleration Longitudinal', array=self.acc_lon_array)
        fasts = S(items=[Section('Fast', slice(10, 40), 10, 40),
                         Section('Fast', slice(71, 74), 71, 74)])

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 0)

    def test_derive_large_delta(self):
        acc_lon = P(name='Acceleration Longitudinal', array=self.acc_lon_array+1.5)

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 1)
        self.assertAlmostEqual(acc_lon_offset_kpv[0].value, 1.505875, delta=0.00000000000001)

    def test_derive_masked_data(self):
        acc_lon = P(name='Acceleration Longitudinal', array=self.acc_lon_array)

        acc_lon.array[10:75] = np.ma.masked

        acc_lon_offset_kpv = AccelerationLongitudinalOffset()
        acc_lon_offset_kpv.derive(acc_lon, self.mobiles, self.fasts)

        self.assertEqual(len(acc_lon_offset_kpv), 0)


class TestAccelerationLongitudinalDuringTakeoffMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalDuringTakeoffMax
        self.operational_combinations = [('Acceleration Longitudinal', 'Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLongitudinalDuringLandingMin(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalDuringLandingMin
        self.operational_combinations = [('Acceleration Longitudinal', 'Landing')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLongitudinalWhileAirborneMax(unittest.TestCase,
                                                   NodeTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalWhileAirborneMax
        self.operational_combinations = [
            ('Acceleration Longitudinal', 'Airborne')
        ]

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.G)
        self.assertEqual(node.name,
                         'Acceleration Longitudinal While Airborne Max')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive(self):
        array = -0.1 + np.ma.sin(np.arange(0, 3.14*2, 0.04))
        accel_long = P(name='Acceleration Longitudinal',
                       array=array)
        airborne = buildsection('Airborne', 5, 150)
        node = self.node_class()
        node.derive(accel_long, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 118)
        self.assertAlmostEqual(node[0].value, -1.1, places=1)


########################################
# Acceleration: Normal


class TestAccelerationNormalMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormalMax
        self.operational_combinations = [('Acceleration Normal', 'Mobile')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationNormal20FtTo5FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormal20FtTo5FtMax
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Acceleration Normal', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 5), {})]

    def test_derive(self):
        '''
        Depends upon DerivedParameterNode.slices_from_to and library.max_value.
        '''
        # Test height range limit:
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(48, 0, -3))
        acc_norm = P('Acceleration Normal', np.ma.concatenate((np.arange(10, 18), np.arange(18, 10, -1))) / 10.0)
        node = AccelerationNormal20FtTo5FtMax()
        node.derive(acc_norm, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(index=10, value=1.6, name='Acceleration Normal 20 Ft To 5 Ft Max'),
        ])
        # Test peak acceleration:
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(32, 0, -2))
        node = AccelerationNormal20FtTo5FtMax()
        node.derive(acc_norm, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(index=8, value=1.8, name='Acceleration Normal 20 Ft To 5 Ft Max'),
        ])


class TestAccelerationNormalWithFlapUpWhileAirborneMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapUpWhileAirborneMax
        self.operational_combinations = [
            ('Acceleration Normal', 'Flap Lever', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever (Synthetic)', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]

    def test_derive(self):
        acc_norm = P(
            name='Acceleration Offset Normal Removed',
            array=np.ma.array(np.tile([0.1, 0.3, -0.1, -0.2, 0.1, 0.2], 10)),
        )
        airborne = buildsection('Airborne', 2, 48)
        name = self.node_class.get_name()

        array = np.ma.repeat((0, 1, 5, 15, 25, 30), 10)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7.0, value=0.3, name=name),
        ]))

        array = np.ma.repeat((1, 2, 5, 15, 25, 30), 10)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7.0, value=0.3, name=name),
        ]))


class TestAccelerationNormalWithFlapUpWhileAirborneMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapUpWhileAirborneMin
        self.operational_combinations = [
            ('Acceleration Normal', 'Flap Lever', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever (Synthetic)', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]

    def test_derive(self):
        acc_norm = P(
            name='Acceleration Offset Normal Removed',
            array=np.ma.array(np.tile([0.1, 0.3, -0.1, -0.2, 0.1, 0.2], 10)),
        )
        airborne = buildsection('Airborne', 2, 48)
        name = self.node_class.get_name()

        array = np.ma.repeat((0, 1, 5, 15, 25, 30), 10)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=9.0, value=-0.2, name=name),
        ]))

        array = np.ma.repeat((1, 2, 5, 15, 25, 30), 10)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=9.0, value=-0.2, name=name),
        ]))


class TestAccelerationNormalWithFlapDownWhileAirborneMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapDownWhileAirborneMax
        self.operational_combinations = [
            ('Acceleration Normal', 'Flap Lever', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever (Synthetic)', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]

    def test_derive(self):
        acc_norm = P(
            name='Acceleration Offset Normal Removed',
            array=np.ma.array(np.tile([0.1, 0.3, -0.1, -0.2, 0.1, 0.2], 5)),
        )
        airborne = buildsection('Airborne', 2, 28)
        name = self.node_class.get_name()

        array = np.ma.repeat((0, 1, 5, 15, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7, value=0.3, name=name),
        ]))

        array = np.ma.repeat((1, 2, 5, 15, 25, 30), 5)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7, value=0.3, name=name),
        ]))


class TestAccelerationNormalWithFlapDownWhileAirborneMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapDownWhileAirborneMin
        self.operational_combinations = [
            ('Acceleration Normal', 'Flap Lever', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever (Synthetic)', 'Airborne'),
            ('Acceleration Normal', 'Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]

    def test_derive(self):
        acc_norm = P(
            name='Acceleration Offset Normal Removed',
            array=np.ma.array(np.tile([0.1, 0.2, -0.1, -0.2, 0.1, 0.2], 5)),
        )
        airborne = buildsection('Airborne', 2, 28)
        name = self.node_class.get_name()

        array = np.ma.repeat((0, 1, 5, 15, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=9, value=-0.2, name=name),
        ]))

        array = np.ma.repeat((1, 2, 5, 15, 25, 30), 5)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(acc_norm, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=9, value=-0.2, name=name),
        ]))


class TestAccelerationNormalAboveLimitWithFlapDownWhileAirborne(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AccelerationNormalAboveLimitWithFlapDownWhileAirborne
        self.operational_combinations = [('Acceleration Normal',
                                          'Acceleration Normal High Limit With Flaps Down',
                                        'Airborne')]

    def test_derive(self):
        array = np.ma.repeat(1.0, 200)
        array[100] = 2.2
        array[130] = 1.8  # On ground
        array[160] = 1.6
        acc_norm = P('Acceleration Normal', array=array)
        acc_limit = P('Acceleration Normal High Limit With Flaps Down',
                      array=np.ma.repeat(2.0, 200))
        airborne = buildsections('Airborne', [30, 120], [150, 180])
        node = self.node_class()
        node.derive(acc_norm, acc_limit, airborne)

        self.assertEqual(len(node), 1)  # Only 1 max per flight
        self.assertEqual(node[0].index, 100)
        self.assertAlmostEqual(node[0].value, 0.2)


class TestAccelerationNormalAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalAtLiftoff
        self.operational_combinations = [('Acceleration Normal', 'Liftoff')]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_norm = Mock()
        liftoffs = KTI('Liftoff', items=[
            KeyTimeInstance(3, 'Liftoff'),
            KeyTimeInstance(1, 'Liftoff'),
        ])
        node = AccelerationNormalAtLiftoff()
        node.derive(acc_norm, liftoffs)
        bump.assert_has_calls([
            call(acc_norm, liftoffs[0].index),
            call(acc_norm, liftoffs[1].index),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Normal At Liftoff'),
            KeyPointValue(1, 2.0, 'Acceleration Normal At Liftoff'),
        ])


class TestAccelerationNormalAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalAtTouchdown
        self.operational_combinations = [
            ('Acceleration Normal', 'Touchdown', 'Landing', 'Landing Roll', 'Touch And Go'),
            ('Acceleration Normal', 'Touchdown', 'Landing', 'Landing Roll', 'Touch And Go'),
            ('Acceleration Normal', 'Touchdown', 'Landing', 'Landing Roll'),
        ]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive_only_touchdowns(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_norm = Mock()
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3, 'Touchdown'),
            KeyTimeInstance(1, 'Touchdown'),
        ])
        ldgs = S('Landing')
        ldgs.create_sections([slice(3, 5, None), slice(1, 3, None)])
        ldg_rolls = S('Landing Roll')
        ldg_rolls.create_sections([slice(3, 5, None), slice(1, 3, None)])
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns, ldgs, ldg_rolls, None)
        bump.assert_has_calls([
            call(acc_norm, touchdowns[0].index, end=5),
            call(acc_norm, touchdowns[1].index, end=3),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Normal At Touchdown'),
            KeyPointValue(1, 2.0, 'Acceleration Normal At Touchdown'),
        ])

    @patch('analysis_engine.key_point_values.bump')
    def test_derive_touchdown_and_touch_and_go(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_norm = Mock()
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(3, 'Touchdown')])
        touch_and_go = KTI('Touch And Go', items=[KeyTimeInstance(1, 'Touch And Go')])
        ldgs = S('Landing')
        ldgs.create_section(slice(3, 6, None))
        ldg_rolls = S('Landing Roll')
        ldg_rolls.create_section(slice(4, 5, None))
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns, ldgs, ldg_rolls, touch_and_go)
        bump.assert_has_calls([
            call(acc_norm, touchdowns[0].index, end=5),
            call(acc_norm, touch_and_go[0].index),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Normal At Touchdown'),
            KeyPointValue(1, 2.0, 'Acceleration Normal At Touchdown'),
        ])

    def test_derive_tdwn_outside_ldg(self):
        acc_norm = P('Acceleration Normal', array=np.ma.arange(25))
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])

        ldgs = S('Landing')
        ldgs.create_section(slice(10, 20, None))
        ldg_rolls = S('Landing Roll')
        ldg_rolls.create_section(slice(21, 25, None))
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns, ldgs, ldg_rolls, None)
        self.assertEqual(node, [
            KeyPointValue(16, 16, 'Acceleration Normal At Touchdown'),
        ])

    def test_derive_ldg_roll_outside_ldg(self):
        acc_norm = Mock()
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(9, 'Touchdown')])

        ldgs = S('Landing')
        ldgs.create_section(slice(10, 20, None))
        ldg_rolls = S('Landing Roll')
        ldg_rolls.create_section(slice(12, 16, None))
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns, ldgs, ldg_rolls, None)
        self.assertEqual(node, [])

    @patch('analysis_engine.key_point_values.bump')
    def test_derive_touch_and_go_no_landing(self, bump):
        bump.side_effect = [(1, 2,), (3, 4)]
        acc_norm = Mock()
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])
        touch_and_go = KTI('Touch And Go', items=[KeyTimeInstance(1, 'Touch And Go')])
        ldgs = S('Landing')
        ldgs.create_section(slice(10, 20, None))
        ldg_rolls = S('Landing Roll')
        ldg_rolls.create_section(slice(21, 25, None))
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns, ldgs, ldg_rolls, touch_and_go)
        bump.assert_has_calls([
            call(acc_norm, touchdowns[0].index, end=None),
            call(acc_norm, touch_and_go[0].index),
        ])
        self.assertEqual(node, [
            KeyPointValue(1, 2.0, 'Acceleration Normal At Touchdown'),
            KeyPointValue(3, 4.0, 'Acceleration Normal At Touchdown'),
        ])


class TestAccelerationNormalMinusLoadFactorThresholdAtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AccelerationNormalMinusLoadFactorThresholdAtTouchdown
        # The following series, model & mods should return 128367KG
        self.model = A('Model', 'B767-232(F)')
        self.series = A('Series', 'B767-200')
        self.mods = A('Modifications', ['Freighter Conversion',])

        self.operational_combinations = [('Acceleration Normal At Touchdown',
                                          'Load Factor Threshold At Touchdown')]
        self.tdwn_idx = 4
        self.gw_under_value_767 = 128367
        self.gw_over_value_767 = 138367

        self.gw_under_value_737 = 55000 # 55112 max ldg wt 737-600
        self.gw_over_value_737 = 56600 # 55112 max ldg wt 737-600

        name = 'Gross Weight'
        array = np.ma.zeros(10)
        array[self.tdwn_idx] = self.gw_under_value_767
        self.gw_under_767 = P(name, array.copy())
        array[self.tdwn_idx] = self.gw_over_value_767
        self.gw_over_767 = P(name, array.copy())

        array[self.tdwn_idx] = self.gw_under_value_737
        self.gw_under_737 = P(name, array.copy())
        array[self.tdwn_idx] = self.gw_over_value_737
        self.gw_over_737 = P(name, array.copy())

        name = 'Acceleration Normal At Touchdown'
        self.land_vert_acc_8hz_767 = KPV(name=name, frequency=8.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.0, name=name),
        ])
        self.land_vert_acc_16hz_767 = KPV(name=name, frequency=16.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.0, name=name),
        ])
        self.land_vert_acc_8hz_737 = KPV(name=name, frequency=8.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.5, name=name),
        ])
        self.land_vert_acc_16hz_737 = KPV(name=name, frequency=16.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.5, name=name),
        ])

        name = 'Gross Weight At Touchdown'
        self.gw_under_kpv_767 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_under_value_767,
                          name=name),
        ])
        self.gw_over_kpv_767 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_over_value_767,
                          name=name),
        ])

        self.gw_under_kpv_737 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_under_value_737,
                          name=name),
        ])

        self.tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(index=self.tdwn_idx, name='Touchdown'),
        ])
        self.touch_and_go = KTI('Touch And Go', items=[])
        self.roll = np.ma.zeros(10)

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.G)
        self.assertEqual(
            node.name,
            'Acceleration Normal Minus Load Factor Threshold At Touchdown'
        )

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(
            model=self.model, series=self.series, mods=self.mods)
        self.assertEqual(len(opts), 1)
        self.assertEqual(opts, self.operational_combinations)

    def _call_derive(self, land_vert_acc, load_factor):
        node = self.node_class()
        node.derive(land_vert_acc=land_vert_acc, load_factors=load_factor)
        self.assertEqual(len(node), 1)
        return node

    def _call_derive_load_factor(self, roll_value, land_vert_acc, gw_kpv=[], gw=None):
        load_factor_kpv = LoadFactorThresholdAtTouchdown()

        self.roll[self.tdwn_idx] = roll_value
        roll = P('Roll', self.roll)

        load_factor_kpv.derive(land_vert_acc=land_vert_acc, roll=roll,
                               tdwns=self.tdwns, gw_kpv=gw_kpv,
                           gw=gw, series=self.series, model=self.model,
                           mods=self.mods, touch_and_go=self.touch_and_go)
        return load_factor_kpv

    def test_derive_767(self):
        # For each weight and frequency, check all seven roll attitudes
        # give increasing margins below a nominal 2g landing.

        # Roll 0-6, weight <= MLW+2500LB @ 16Hz
        expected_val = [0.10, 0.10, 0.10, 0.21, 0.33, 0.44, 0.55]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_767, gw_kpv=self.gw_under_kpv_767)
            node = self._call_derive(self.land_vert_acc_16hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+2500LB @ 16Hz
        expected_val = [0.45, 0.45, 0.50, 0.55, 0.61, 0.66, 0.71]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_767, gw_kpv=self.gw_over_kpv_767)
            node = self._call_derive(self.land_vert_acc_16hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight <= MLW+2500LB @ 8Hz
        expected_val = [0.20, 0.20, 0.20, 0.30, 0.40, 0.50, 0.60]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_767, gw_kpv=self.gw_under_kpv_767)
            node = self._call_derive(self.land_vert_acc_8hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+2500LB @ 8Hz
        expected_val = [0.50, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_767, gw_kpv=self.gw_over_kpv_767)
            node = self._call_derive(self.land_vert_acc_8hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Use weight from 'Gross Weight' instead of from KPV
        # Roll 0-6, weight <= MLW+2500LB @ 16Hz
        expected_val = [0.10, 0.10, 0.10, 0.21, 0.33, 0.44, 0.55]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_767, gw=self.gw_under_767)
            node = self._call_derive(self.land_vert_acc_16hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+2500LB @ 16Hz
        expected_val = [0.45, 0.45, 0.50, 0.55, 0.61, 0.66, 0.71]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_767, gw=self.gw_over_767)
            node = self._call_derive(self.land_vert_acc_16hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight <= MLW+2500LB @ 8Hz
        expected_val = [0.20, 0.20, 0.20, 0.30, 0.40, 0.50, 0.60]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_767, gw=self.gw_under_767)
            node = self._call_derive(self.land_vert_acc_8hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+2500LB @ 8Hz
        expected_val = [0.50, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_767, gw=self.gw_over_767)
            node = self._call_derive(self.land_vert_acc_8hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+2500LB @ 8Hz
        expected_val = [0.50, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        self.touch_and_go = KTI('Touch And Go', items=[KeyTimeInstance(index=self.tdwn_idx, name='Touch And Go')])
        self.tdwns = KTI('Touchdown', items=[])
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_767, gw=self.gw_over_767)
            node = self._call_derive(self.land_vert_acc_8hz_767, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

    def test_derive_737(self):

        # ==========================
        # Boeing 737 family tests
        # ==========================

        self.series = A('Series', 'B737-600')
        self.model = A('Model', 'B737-683')
        self.mods = A('Modifications', ['N/A',])

        # Roll 0-6, weight <= MLW+1000LB @ 8Hz
        expected_val = [0.40, 0.40, 0.40, 0.5375, 0.675, 0.8125, 0.95]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_737, gw=self.gw_under_737)
            node = self._call_derive(self.land_vert_acc_8hz_737, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight <= MLW+1000LB @ 16Hz
        expected_val = [0.30, 0.30, 0.30, 0.45, 0.6, 0.75, 0.9]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_737, gw=self.gw_under_737)
            node = self._call_derive(self.land_vert_acc_16hz_737, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+1000LB @ 8Hz
        expected_val = [0.70, 0.70, 0.78, 0.86, 0.94, 1.02, 1.1]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_8hz_737, gw=self.gw_over_737)
            node = self._call_derive(self.land_vert_acc_8hz_737, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

        # Roll 0-6, weight > MLW+1000LB @ 16Hz
        expected_val = [0.65, 0.65, 0.73, 0.82, 0.90, 0.99, 1.07]
        for idx, roll in enumerate(np.arange(0.0, 7.0)):
            load_factor = self._call_derive_load_factor(roll, self.land_vert_acc_16hz_737, gw=self.gw_over_737)
            node = self._call_derive(self.land_vert_acc_16hz_737, load_factor)[0]
            self.assertAlmostEqual(node.value, expected_val[idx], places=2)

    def test_roll_peak_767(self):
        roll = P('Roll', array=np.array(
            np.linspace(-8.0, 2.0, num=40, endpoint=False, dtype=float)),
                 offset=0.0, frequency=4.0)

        load_factor = LoadFactorThresholdAtTouchdown()
        load_factor.derive(land_vert_acc=self.land_vert_acc_16hz_767, roll=roll,
                           tdwns=self.tdwns, gw_kpv=self.gw_under_kpv_767,
                           gw=self.gw_under_767,
                           series=self.series, model=self.model, mods=self.mods,
                           touch_and_go=self.touch_and_go)

        node = self.node_class()
        node.derive(land_vert_acc=self.land_vert_acc_16hz_767, load_factors=load_factor)

        # The result corresponds to 5deg (at 3 sec) not 4deg (at 4 sec) as we are
        # looking across the preceding second.
        self.assertAlmostEqual(node[0].value, 0.44, places=2)

    def test_roll_peak_737(self):
        self.series = A('Series', 'B737-600')
        self.model = A('Model', 'B737-683')
        self.mods = A('Modifications', ['N/A',])
        roll = P('Roll', array=np.array(
            np.linspace(-7.0, 3.0, num=40, endpoint=False, dtype=float)),
                 offset=0.0, frequency=4.0)

        load_factor = LoadFactorThresholdAtTouchdown()
        load_factor.derive(land_vert_acc=self.land_vert_acc_16hz_737, roll=roll,
                           tdwns=self.tdwns, gw_kpv=[],
                           gw=self.gw_over_737,
                           series=self.series, model=self.model, mods=self.mods,
                           touch_and_go=self.touch_and_go)

        node = self.node_class()
        node.derive(land_vert_acc=self.land_vert_acc_16hz_737, load_factors=load_factor)

        self.assertAlmostEqual(node[0].value, 0.90, places=2)

    def test_roll_over_6_deg_767(self):
        roll = P('Roll', array=np.array([8.0]*10))

        load_factor = LoadFactorThresholdAtTouchdown()
        load_factor.derive(land_vert_acc=self.land_vert_acc_16hz_767, roll=roll,
                           tdwns=self.tdwns, gw_kpv=self.gw_under_kpv_767,
                           gw=self.gw_under_767,
                           series=self.series, model=self.model, mods=self.mods,
                           touch_and_go=self.touch_and_go)

        node = self.node_class()
        node.derive(land_vert_acc=self.land_vert_acc_16hz_767, load_factors=load_factor)

        self.assertAlmostEqual(node[0].value, 2.0, places=2)


class TestLoadFactorThresholdAtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = LoadFactorThresholdAtTouchdown
        # The following series, model & mods should return 128367KG
        self.model = A('Model', 'B767-232(F)')
        self.series = A('Series', 'B767-200')
        self.mods = A('Modifications', ['Freighter Conversion',])

        self.operational_combinations = [('Acceleration Normal At Touchdown',
                                          'Roll', 'Touchdown',
                                         'Gross Weight At Touchdown',
                                         'Gross Weight',
                                         'Model',
                                         'Series',
                                         'Modifications',
                                         'Touch And Go')]
        self.tdwn_idx = 4
        self.gw_under_value_767 = 128367
        self.gw_over_value_767 = 138367

        self.gw_under_value_737 = 55000 # 55112 max ldg wt 737-600
        self.gw_over_value_737 = 56600 # 55112 max ldg wt 737-600

        name = 'Gross Weight'
        array = [0]*10
        array[self.tdwn_idx] = self.gw_under_value_767
        self.gw_under_767 = P(name, np.ma.array(array))
        array[self.tdwn_idx] = self.gw_over_value_767
        self.gw_over_767 = P(name, np.ma.array(array))

        array[self.tdwn_idx] = self.gw_under_value_737
        self.gw_under_737 = P(name, np.ma.array(array))
        array[self.tdwn_idx] = self.gw_over_value_737
        self.gw_over_737 = P(name, np.ma.array(array))

        name = 'Acceleration Normal At Touchdown'
        self.land_vert_acc_8hz_767 = KPV(name=name, frequency=8.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.0, name=name),
        ])
        self.land_vert_acc_16hz_767 = KPV(name=name, frequency=16.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.0, name=name),
        ])
        self.land_vert_acc_8hz_737 = KPV(name=name, frequency=8.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.5, name=name),
        ])
        self.land_vert_acc_16hz_737 = KPV(name=name, frequency=16.0, items=[
            KeyPointValue(index=self.tdwn_idx, value=2.5, name=name),
        ])

        name = 'Gross Weight At Touchdown'
        self.gw_under_kpv_767 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_under_value_767,
                          name=name),
        ])
        self.gw_over_kpv_767 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_over_value_767,
                          name=name),
        ])

        self.gw_under_kpv_737 = KPV(name=name, items=[
            KeyPointValue(index=self.tdwn_idx, value=self.gw_under_value_737,
                          name=name),
        ])

        self.tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(index=self.tdwn_idx, name='Touchdown'),
        ])
        self.touch_and_go = KTI('Touch And Go', items=[])
        self.roll = np.ma.zeros(10)

    def _test_landing_weight(self, series, model, mods, weight):
        self.assertEqual(
            self.node_class.get_landing_weight(series, model, mods),
            weight
        )

    def test_get_landing_weight(self):
        ac_variations=[
            [self.series.value, self.model.value, self.mods.value, 128367],
            ['B757-200', 'B757-236(F)', 'MOD Aircraft Line Numbers 1-209', 89992],
            ['B757-200', 'B757-24A(PF)', 'MOD Aircraft Line Numbers 1-209', 89992],
            ['B757-200', 'B757-200(PCF)', 'MOD Aircraft Line Numbers 1-209', 89811],
            ['B757-200', 'B757-233(F)', 'N/A', 95254],
            ['B757-200', 'B757-24A(PF)', 'N/A', 95254],
            ['B757-200', 'B757-236(PCF)', 'N/A', 95254],
            ['B757-200', 'N/A', 'MOD Aircraft Line Numbers 210 and above', 95254],
            ['B757-200', 'N/A', 'N/A', 89992],
            ['B757-300', 'N/A', 'N/A', 101605],
            ['B767-200', 'N/A', 'Post TC', 136078],
            ['B767-25E', 'N/A', 'Freighter Conversion', 126098],
            ['B767-200', 'N/A', 'Freighter Conversion', 128367],
            ['B767-200', 'N/A', 'N/A', 123377],
            ['B767-300', 'N/A', 'Freighter Conversion', 147871],
            ['B767-300', 'N/A', 'N/A', 145149],
            ['B767-400', '(ER)', 'N/A', 158757],

            ['B737-600', 'N/A', 'N/A', 55205],
            ['B737-700', 'N/A', 'N/A', 59190],
            ['B737-700', 'N/A', 'Increased Gross Weight', 61389],
            ['B737-800', 'N/A', 'N/A', 65970],
            ['B737-800', 'N/A', 'Increased Gross Weight', 67025],
            ['B737-900', 'N/A', 'N/A', 67482],
            ['B737-900', '(ER)', 'N/A', 72064],
            ['B737-MAX-8', 'N/A', 'N/A', 70003],

            ['N/A', 'N/A', 'N/A', None],
        ]
        for aircraft in ac_variations:
            self._test_landing_weight(*aircraft)

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.G)
        self.assertEqual(
            node.name,
            'Load Factor Threshold At Touchdown'
        )

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(
            model=self.model, series=self.series, mods=self.mods)
        self.assertEqual(len(opts), 1)
        self.assertEqual(opts, self.operational_combinations)

    def _call_derive(self, roll_value, land_vert_acc, gw_kpv=[], gw=None):
        node = self.node_class()

        self.roll[self.tdwn_idx] = roll_value
        roll = P('Roll', self.roll)

        node.derive(land_vert_acc=land_vert_acc, roll=roll,
                    tdwns=self.tdwns, gw_kpv=gw_kpv,
                    gw=gw, series=self.series, model=self.model,
                    mods=self.mods, touch_and_go=self.touch_and_go)

        self.assertEqual(len(node), 1)
        return node


class TestAccelerationNormalAboveWeightLimitAtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AccelerationNormalAboveWeightLimitAtTouchdown

    def test_derive(self):
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight', np.ma.zeros(15))

        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=0, value=0),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)

    def test_derive_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.1)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=4, value=2.50),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 4)
        self.assertAlmostEqual(node[0].value, 0.75)
        self.assertGreater(node[0].value, 0)

    def test_derive_no_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.1)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=13, value=1.95),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 13)
        self.assertAlmostEqual(node[0].value, -0.15)
        self.assertLess(node[0].value, 0)

    def test_derive_hard_touch_and_go_plus_hard_touchdown(self):
        # Both hard landings
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.1)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=3, value=1.95),
            KeyPointValue(index=13, value=2.50),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 3)
        self.assertAlmostEqual(node[0].value, 0.2)
        self.assertEqual(node[1].index, 13)
        self.assertAlmostEqual(node[1].value, 0.4)
        self.assertGreater(node[0].value, 0)
        self.assertGreater(node[1].value, 0)

    def test_derive_hard_touch_and_go_plus_touchdown(self):
        # One hard landing
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight',
                  np.ma.array([1.75, 1.75, 1.75, 1.75, 1.75,
                                 2.00, 2.00, 2.00, 2.00, 2.00,
                                 2.00, 2.00, 2.10, 2.10, 2.10, ]))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=3, value=2.00),
            KeyPointValue(index=13, value=1.90),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 3)
        self.assertAlmostEqual(node[0].value, 0.25)
        self.assertEqual(node[1].index, 13)
        self.assertAlmostEqual(node[1].value, -0.2)
        self.assertGreater(node[0].value, 0)
        self.assertLess(node[1].value, 0)

    def test_derive_touch_and_go_plus_touchdown(self):
        # No hard landings
        node = self.node_class()
        limit = P('Acceleration Normal Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.1)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=3, value=1.29),
            KeyPointValue(index=13, value=1.20),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 3)
        self.assertAlmostEqual(node[0].value, -0.46)
        self.assertEqual(node[1].index, 13)
        self.assertAlmostEqual(node[1].value, -0.9)
        self.assertLess(node[0].value, 0)
        self.assertLess(node[1].value, 0)

class TestAccelerationNormalAboveWeightLowLimitAtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AccelerationNormalAboveWeightLowLimitAtTouchdown

    def test_derive(self):
        node = self.node_class()
        limit = P('Acceleration Normal Low Limit For Landing Weight', np.ma.zeros(15))

        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=0, value=0),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)

    def test_derive_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal Low Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.2)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=4, value=2.50),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 4)
        self.assertAlmostEqual(node[0].value, 0.75)
        self.assertGreater(node[0].value, 0)

    def test_derive_no_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal Low Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.75, np.ones(7) * 2.0, np.ones(3) * 2.2)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=13, value=1.95),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 13)
        self.assertAlmostEqual(node[0].value, -0.25)
        self.assertLess(node[0].value, 0)

class TestAccelerationNormalAboveWeightHighLimitAtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AccelerationNormalAboveWeightHighLimitAtTouchdown

    def test_derive(self):
        node = self.node_class()
        limit = P('Acceleration Normal High Limit For Landing Weight', np.ma.zeros(15))

        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=0, value=0),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)

    def test_derive_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal High Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.93, np.ones(7) * 2.2, np.ones(3) * 2.42)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=4, value=2.50),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 4)
        self.assertAlmostEqual(node[0].value, 0.57)
        self.assertGreater(node[0].value, 0)

    def test_derive_no_hard_landing(self):
        node = self.node_class()
        limit = P('Acceleration Normal High Limit For Landing Weight',
                  np.ma.concatenate((np.ones(5) * 1.93, np.ones(7) * 2.2, np.ones(3) * 2.42)))
        acc_norm_tdwns = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(index=13, value=1.95),
        ])
        node.derive(acc_norm_tdwns=acc_norm_tdwns,
                    acc_limit=limit)

        self.assertEqual(node[0].index, 13)
        self.assertAlmostEqual(node[0].value, -0.47)
        self.assertLess(node[0].value, 0)

class TestAccelerationNormalLiftoffTo35FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalLiftoffTo35FtMax
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Acceleration Normal', 'Liftoff', 'Takeoff')]
        self.function = max_value

    def test_derive(self):
        acc_norm = P(
            name='Acceleration Offset Normal Removed',
            array=np.ma.array(np.tile([0.05, 0.1, -0.2, -0.1, 0.2, 0.1], 5)),
        )
        liftoffs = KTI('Liftoff', items=[
            KeyTimeInstance(5.5, 'Liftoff'),
        ])
        toff = buildsection('Takeoff', 2,9)
        node = self.node_class()
        node.derive(acc_norm, liftoffs, toff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 7)
        self.assertEqual(node[0].value, 0.1)


class TestAccelerationNormalOffset(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalOffset
        self.operational_combinations = [('Acceleration Normal', 'Taxiing')]
        self.acc_normal_array=np.ma.array([0] * 17 + [0.02, 0.05, 0.02, 0, -0.017,] + [0] * 7 +
                                          [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 6 + [-0.02] +
                                          [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 10 + [0.02, 0.04, 0.01] + [0.011] * 4 + [0] * 10)*1.5
        self.taxiing = S(items=[Section('Taxiing', slice(10, 30), 10, 30),
                                Section('Taxiing', slice(60, 75), 60, 75)])

    def test_derive_normal_scenario(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array)

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, 0.00887142857143, delta=0.00000000000001)

    def test_derive_negative_values(self):
        acc_normal = P(name='Acceleration Longitudinal', array=-self.acc_normal_array)

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, -0.00887142857143, delta=0.00000000000001)

    def test_derive_all_values_equal_0(self):
        acc_normal = P(name='Acceleration Longitudinal', array=np.zeros_like(self.acc_normal_array))

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, 0.0, delta=0.00000000000001)

    def test_derive_values_in_given_slices_all_0(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array)
        acc_normal.array[10:30] = acc_normal.array[60:75] = 0.0

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, 0.0, delta=0.00000000000001)

    def test_derive_values_in_one_slice_only(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array)
        acc_normal.array[60:75] = 0.0

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, 0.00398571428571, delta=0.00000000000001)

    def test_derive_not_enough_samples(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array)
        taxiing = S(items=[Section('Taxiing', slice(10, 15), 10, 15)])

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 0)

    def test_derive_large_delta(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array+1.5)

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 1)
        self.assertAlmostEqual(acc_normal_offset_kpv[0].value, 1.50887142857, delta=0.0000000001)

    def test_derive_masked_data(self):
        acc_normal = P(name='Acceleration Longitudinal', array=self.acc_normal_array)
        acc_normal.array[10:30] = acc_normal.array[60:75] = np.ma.masked

        acc_normal_offset_kpv = AccelerationNormalOffset()
        acc_normal_offset_kpv.derive(acc_normal, self.taxiing)

        self.assertEqual(len(acc_normal_offset_kpv), 0)


class TestAccelerationNormalWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationNormalWhileAirborneMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Normal', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 400)
        accel_lat = P(
            name='Acceleration Normal',
            array=-x*np.sin(x),
            frequency=4)
        name = 'Airborne'
        section = Section(name, slice(0, 300), 0, 300)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(accel_lat, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 196)
        self.assertAlmostEqual(node[0].value, 4.814, places=3)


class TestAccelerationNormalWhileAirborneMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationNormalWhileAirborneMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Normal', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 400)
        accel_lat = P(
            name='Acceleration Normal',
            array=-x*np.sin(x),
            frequency=4
        )
        name = 'Airborne'
        section = Section(name, slice(0, 300), 0, 300)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(accel_lat, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 299)
        self.assertAlmostEqual(node[0].value, -7.013, places=3)


##############################################################################
# Airspeed


########################################
# Airspeed: General


class TestAirspeedAt8000FtDescending(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AirspeedAt8000FtDescending
        self.operational_combinations = [('Airspeed', 'Altitude When Descending')]

    def test_derive_basic(self):
        air_spd = P('Airspeed', array=np.ma.arange(0, 200, 10))
        alt_std_desc = AltitudeWhenDescending(
            items=[KeyTimeInstance(8, '6000 Ft Descending'),
                   KeyTimeInstance(10, '8000 Ft Descending'),
                   KeyTimeInstance(16, '8000 Ft Descending'),
                   KeyTimeInstance(18, '8000 Ft Descending')])
        node = self.node_class()
        node.derive(air_spd, alt_std_desc)
        self.assertEqual(node,
                         [KeyPointValue(index=10, value=100.0, name='Airspeed At 8000 Ft Descending'),
                          KeyPointValue(index=16, value=160.0, name='Airspeed At 8000 Ft Descending'),
                          KeyPointValue(index=18, value=180.0, name='Airspeed At 8000 Ft Descending')])


class TestAirspeedMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMax
        self.operational_combinations = [('Airspeed', 'Airborne')]
        self.function = max_value

    def test_derive_basic(self):
        array = np.ma.cos(np.arange(0, 12.6, 0.1)) * -100 + 100
        kpv = AirspeedMax()
        kpv.derive(
            P('Airspeed', array),
            [Section('Airborne', w, w.start, w.stop) for w in runs_of_ones(array >= 80)],
        )
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 31)
        self.assertGreater(kpv[0].value, 199.9)
        self.assertLess(kpv[0].value, 200)
        self.assertEqual(kpv[1].index, 94)
        self.assertGreater(kpv[1].value, 199.9)
        self.assertLess(kpv[1].value, 200)


class TestAirspeedDuringCruiseMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringCruiseMax
        self.operational_combinations = [('Airspeed', 'Cruise')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedDuringCruiseMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringCruiseMin
        self.operational_combinations = [('Airspeed', 'Cruise')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedGustsDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedGustsDuringFinalApproach
        self.operational_combinations = [('Airspeed', 'Groundspeed', 'Altitude Radio', 'Airborne')]

    def test_derive_basic(self):
        # This function interpolates twice, hence the more complex test case.
        air_spd = P(
            name='Airspeed',
            array=np.ma.array([180, 180, 180, 180, 170, 150, 140, 120, 100]),
            frequency=1.0,
            offset=0.0,
        )
        gnd_spd = P(
            name='Groundspeed',
            array=np.ma.array([180, 180, 180, 180, 170, 100, 100, 100, 100]),
            frequency=1.0,
            offset=0.0,
        )
        alt_rad = P(
            name='Altitude Radio',
            array=np.ma.array([45, 45, 45, 45, 35, 25, 15, 5, 0]),
            frequency=1.0,
            offset=0.0,
        )
        airborne = S(items=[Section('Airborne', slice(3, 9), 3, 9)])
        kpv = AirspeedGustsDuringFinalApproach()
        kpv.get_derived([air_spd, gnd_spd, alt_rad, airborne])
        self.assertEqual(kpv[0].value, 25)
        self.assertEqual(kpv[0].index, 4.75)


########################################
# Airspeed: Climbing


class TestAirspeedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtLiftoff
        self.operational_combinations = [('Airspeed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAt35FtDuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAt35FtDuringTakeoff
        self.operational_combinations = [('Airspeed', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeed35To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')

class TestAirspeed1000To5000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed1000To5000FtMax
        self.operational_combinations = [
            ('Airspeed', 'Altitude AAL For Flight Phases', 'Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed35To1000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To8000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed1000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases',
                                          'Altitude STD Smoothed', 'Climb')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 8000), {})]

    def test_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_aal = P('Altitude AAL For Flight Phases', array * 50)
        alt_std = P('Altitude STD Smoothed', array * 50 + 2000)
        climb = buildsections('Climb', [3, 28], [65, 91])
        event = Airspeed1000To8000FtMax()
        event.derive(spd, alt_aal, alt_std, climb)
        self.assertEqual(event[0].index, 17)
        self.assertAlmostEqual(event[0].value, 112.88, 1)
        self.assertEqual(event[1].index, 80.0)
        self.assertAlmostEqual(event[1].value, 114.55, 1)


class TestAirspeed5000To8000FtMax(unittest.TestCase):
    def setUp(self):
        self.node_class = Airspeed5000To8000FtMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.KT)
        self.assertEqual(node.name, 'Airspeed 5000 To 8000 Ft Max')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertIn('Airspeed', opts[0])
        self.assertIn('Altitude AAL For Flight Phases', opts[0])
        self.assertIn('Altitude STD Smoothed', opts[0])
        self.assertIn('Climb', opts[0])

    def test_derive(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        air_spd = P('Airspeed', array)
        alt_aal = P('Altitude AAL For Flight Phases', array * 50+2000)
        alt_std = P('Altitude STD Smoothed', array * 50 + 2000)
        climb = buildsections('Climb', [3, 28], [65, 91])
        node = self.node_class()
        node.derive(air_spd, alt_aal, alt_std, climb)
        self.assertEqual(node[0].index, 17)
        self.assertAlmostEqual(node[0].value, 112.88, 1)
        self.assertEqual(node[1].index, 80)
        self.assertAlmostEqual(node[1].value, 114.55, 1)


class TestAirspeed5000To10000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed5000To10000FtMax
        self.operational_combinations = [
            ('Airspeed', 'Altitude AAL For Flight Phases',
             'Altitude STD Smoothed', 'Climb')
        ]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed8000To10000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed8000To10000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Descending


class TestAirspeed10000To8000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed10000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Descent')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed8000To5000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Airspeed8000To5000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Altitude STD Smoothed', 'Descent')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (8000, 5000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAboveStickShakerSpeedMin(unittest.TestCase):

    def setUp(self):
        self.node_class=AirspeedAboveStickShakerSpeedMin

    def test_derive(self):
        airspeed_array = np.ma.arange(50, 250, 1)
        shaker_array = np.ma.array([150]*120 + [160]*30 + [210]*50)
        airborne = buildsection('Airborne', 105, 200)
        ias = P('Airspeed', airspeed_array)
        shaker = P('Stick Shaker Speed', shaker_array)
        node = self.node_class()
        node.derive(ias, shaker, airborne)
        self.assertEqual(node[0].value, -10)
        self.assertEqual(node[0].index, 150)


class TestAirspeed10000To5000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Airspeed10000To5000FtMax
        self.operational_combinations = [
            ('Airspeed', 'Altitude AAL For Flight Phases',
             'Altitude STD Smoothed', 'Descent')
        ]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 5000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed5000To3000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed5000To3000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Descent')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed3000To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed3000To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMax

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate, [('Airspeed', 'Altitude AAL For Flight Phases', 'Final Approach')])

    def test_derive_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_ph = P('Altitude AAL For Flight Phases', array * 10)
        final_app = buildsections('Final Approach', [47, 60], [109, 123])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertEqual(kpv[0].value, 91.250101656055278)
        self.assertEqual(kpv[1].index, 110)
        self.assertEqual(kpv[1].value, 99.557430201194919)


class TestAirspeed1000To500FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed500To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed500To20FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed500To20FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed500To50FtMedian(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed500To50FtMedian
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = median_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    def test_derive(self):
        aspd = P('Airspeed', array=np.ma.array([999, 999, 999, 999, 999,
                                                145.25,  145.  ,  145.5 ,  146.  ,  146.  ,  145.75,  145.25,
            145.5 ,  145.  ,  147.5 ,  145.75,  145.25,  145.5 ,  145.75,
            145.5 ,  145.25,  142.25,  145.  ,  145.  ,  144.25,  145.75,
            144.5 ,  146.75,  145.75,  146.25,  145.5 ,  144.  ,  142.5 ,
            143.5 ,  144.5 ,  144.  ,  145.5 ,  144.5 ,  143.  ,  142.75,
            999, 999, 999, 999, 999]))
        alt = P('Altitude AAL', array=[502]*5 + list(range(399, 50, -10)) + [0]*5)
        node = Airspeed500To50FtMedian()
        node.derive(aspd, alt)
        self.assertEqual(node,
                         [KeyPointValue(22, 145.25,
                                        name='Airspeed 500 To 50 Ft Median')])


class TestAirspeed500To50FtMedianMinusAirspeedSelected(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Airspeed500To50FtMedianMinusAirspeedSelected
        self.operational_combinations = [('Airspeed Selected',
                                          'Airspeed 500 To 50 Ft Median')]
        self.assertEqual(self.node_class.get_name(),
                         'Airspeed 500 To 50 Ft Median Minus Airspeed Selected')

    def test_derive(self):

        select_spd = P('Airspeed Selected', array=np.ma.ones(35) * 145.0)
        median_spd = Airspeed500To50FtMedian()
        median_spd.create_kpv(10, 145.25)
        diff = Airspeed500To50FtMedianMinusAirspeedSelected()
        diff.derive(select_spd, median_spd)
        self.assertEqual(diff, [KeyPointValue(10, .25,
                                              name='Airspeed 500 To 50 Ft Median Minus Airspeed Selected')])


class TestAirspeedAtAPUpperModesEngaged(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedAtAPUpperModesEngaged

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Airspeed At AP Upper Modes Engaged')
        self.assertEqual(node.units, ut.KT)

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=helicopter), [])
        opts = self.node_class.get_operational_combinations(
            ac_type=helicopter, family=A('Family', 'S92'))
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 10)
        self.assertIn('Airspeed', opts[0])
        self.assertIn('AP (1) Heading Selected Mode Engaged', opts[0])
        self.assertIn('AP (2) Heading Selected Mode Engaged', opts[0])
        self.assertIn('AP (1) Altitude Preselect Mode Engaged', opts[0])
        self.assertIn('AP (2) Altitude Preselect Mode Engaged', opts[0])
        self.assertIn('AP (1) Vertical Speed Mode Engaged', opts[0])
        self.assertIn('AP (2) Vertical Speed Mode Engaged', opts[0])
        self.assertIn('AP (1) Airspeed Mode Engaged', opts[0])
        self.assertIn('AP (2) Airspeed Mode Engaged', opts[0])
        self.assertIn('Initial Climb', opts[0])

    def test_derive(self):
        a = np.concatenate((np.linspace(5,100,13), np.linspace(100,5,17)))
        air_spd = P('Airspeed', np.ma.concatenate((a, a)))
        climb = buildsections('Initial Climb', [1, 10], [31, 40])

        array = np.ma.concatenate((np.zeros(34), np.ones(10), np.zeros(16)))
        ap_1_hdg = M('AP (1) Heading Selected Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_1_alt = M('AP (1) Altitude Preselect Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_1_vrt = M('AP (1) Vertical Speed Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_1_air = M('AP (1) Airspeed Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})

        array = np.ma.concatenate((np.zeros(3), np.ones(11), np.zeros(46)))
        ap_2_hdg = M('AP (2) Heading Selected Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_2_alt = M('AP (2) Altitude Preselect Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_2_vrt = M('AP (2) Vertical Speed Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})
        ap_2_air = M('AP (2) Airspeed Mode Engaged',
                     array,
                     values_mapping={0: '-', 1: 'Engaged'})

        node = self.node_class()
        node.derive(air_spd, ap_1_hdg, ap_2_hdg, ap_1_alt, ap_2_alt,
                    ap_1_vrt, ap_2_vrt, ap_1_air, ap_2_air, climb)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 3)
        self.assertAlmostEqual(node[0].value, 28.75, places=2)
        self.assertEqual(node[1].index, 34)
        self.assertAlmostEqual(node[1].value, 36.67, places=2)

class TestAirspeedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtTouchdown
        self.operational_combinations = [('Airspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinsToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AirspeedMinsToTouchdown
        self.operational_combinations = [('Airspeed', 'Mins To Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTrueAtTouchdown(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedTrueAtTouchdown
        self.operational_combinations = [('Airspeed True', 'Touchdown')]
        air_spd_array = np.ma.array([122, 122, 121, 118, 116, 110, 90, 80, 70, 0, 0])
        self.air_spd = P('Airspeed True', array = air_spd_array)
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=5.2),
        ])

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_low_frequency(self):
        # created based on real Airspeed True recorded at 0.25hz
        self.air_spd.array[6:] = 0
        node = self.node_class()
        node.derive(self.air_spd, self.touchdowns)

        self.assertEqual(len(node), 1)

        self.assertEqual(node[0].index, 5.2)
        self.assertEqual(node[0].value, 110)

    def test_derive_basic(self):
        # created based on real Airspeed True recorded at 0.25hz
        node = self.node_class()
        node.derive(self.air_spd, self.touchdowns)

        self.assertEqual(len(node), 1)

        self.assertEqual(node[0].index, 5.2)
        self.assertEqual(node[0].value, 106)


########################################
# Airspeed: Minus V2


class TestV2AtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = V2AtLiftoff
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=269),
            KeyTimeInstance(name='Liftoff', index=860),
        ])

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('AFR V2', 'Liftoff'),
            afr_v2=A('AFR V2', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('AFR V2', 'Liftoff'),
            afr_v2=A('AFR V2', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('V2-Vac', 'Liftoff'),
        ))
        # Airbus:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed Selected', 'Speed Control', 'Liftoff', 'Manufacturer'),
            manufacturer=A('Manufacturer', 'Airbus'),
        ))
        # V2:
        self.assertTrue(self.node_class.can_operate(
            ('V2','Liftoff')
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(None, None, None, None, self.liftoffs, None)
        self.assertEqual(len(node), 0)

    def test_derive__afr_v2(self):
        afr_v2 = A('AFR V2', 120)
        node = self.node_class()
        node.derive(None, None, None, afr_v2, self.liftoffs, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 120)

    def test_derive__airbus(self):
        manufacturer = A(name='Manufacturer', value='Airbus')
        spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170), (10, 630, 640)))
        spd_sel.array[:10] = np.ma.masked
        node = self.node_class()
        node.derive(None, None, spd_sel, None, self.liftoffs, manufacturer)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 120)

    def test_derive__embraer(self):
        manufacturer = A(name='Manufacturer', value='Embraer')
        v2_vac = P('V2-Vac', np.ma.repeat(150, 1280))
        node = self.node_class()
        node.derive(None, v2_vac, None, None, self.liftoffs, manufacturer)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 150)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 150)

    def test_derive__v2(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        v2 = P(' V2', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        node = self.node_class()
        node.derive(v2, None, None, None, self.liftoffs, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 170)

    def test_derive__v2_superframe(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=269/64.0),
            KeyTimeInstance(name='Liftoff', index=860/64.0),
        ])
        v2 = P(' V2', np.ma.repeat((400, 120, 170, 400, 170), (190/64.0, 130/64.0, 192/64.0, 192/64.0, 448/64.0)), frequency=1/64.0)
        node = self.node_class()
        node.derive(v2, None, None, None, liftoffs, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269//64.0)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860//64.0)
        self.assertEqual(node[1].value, 170)

    def test_derive__v2_masked(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        v2 = P(' V2', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        v2.array[267:272] = np.ma.masked
        node = self.node_class()
        node.derive(v2, None, None, None, self.liftoffs, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 267)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 170)

    def test_derive__fully_masked(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        v2 = P('V2', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        # fully mask first liftoff airspeed selected
        v2.array[0:353] = np.ma.masked
        node = self.node_class()
        node.derive(v2, None, None, None, self.liftoffs, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 860)
        self.assertEqual(node[0].value, 170)

class TestV2LookupAtLiftoff(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined V2.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'v2': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
            'Lever 1': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'v2': {'Lever 2': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'v2': {'Lever 1': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'v2': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
            '15': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'v2': {'5': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'v2': {'17.5': 135}}

    def setUp(self):
        self.node_class = V2LookupAtLiftoff
        self.weight = KPV(name='Gross Weight At Liftoff', items=[
            KeyPointValue(name='Gross Weight At Liftoff', index=20, value=54192.06),
            KeyPointValue(name='Gross Weight At Liftoff', index=860, value=44192.06),
        ])
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=20),
            KeyTimeInstance(name='Liftoff', index=860),
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=420),
            KeyTimeInstance(name='Climb Start', index=1060),
        ])

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Liftoff', 'Climb Start',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(zip(keys, self.generate_attributes('airbus')))
        boeing = dict(zip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(zip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight At Liftoff')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(15, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.weight,
                    self.liftoffs, self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 20)
        self.assertEqual(node[0].value, 138)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 125)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.weight,
                    self.liftoffs, self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 20)
        self.assertEqual(node[0].value, 140)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 140)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, None, self.liftoffs,
                    self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        self.assertEqual(len(node), 0)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, None, self.liftoffs,
                    self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 20)
        self.assertEqual(node[0].value, 135)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 135)


class TestAirspeedSelectedAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedSelectedAtLiftoff
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=269),
            KeyTimeInstance(name='Liftoff', index=860),
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=352),
            KeyTimeInstance(name='Climb Start', index=1060),
        ])
        self.tkof_accel_starts = KTI(name='Takeoff Acceleration Start', items=[
            KeyTimeInstance(name='Takeoff Acceleration Start', index=209),
        ])

    def test_can_operate(self):
        self.assertEqual(self.node_class().get_operational_combinations(),
                         [('Airspeed Selected', 'Liftoff', 'Takeoff Acceleration Start', 'Climb Start')])

    def test_derive(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        spd_sel = P(' Airspeed Selected', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        node = self.node_class()
        node.derive(spd_sel, self.liftoffs, self.tkof_accel_starts, self.climbs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 170)

    def test_derive__superframe(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=269/64.0),
            KeyTimeInstance(name='Liftoff', index=860/64.0),
        ])
        climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=352/64.0),
            KeyTimeInstance(name='Climb Start', index=1060/64.0),
        ])
        tkof_accel_starts = KTI(name='Takeoff Acceleration Start', items=[
            KeyTimeInstance(name='Takeoff Acceleration Start', index=209/64.0),
        ])
        spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448))[::64], frequency=1/64.0)
        node = self.node_class(frequency=1/64.0)
        node.derive(spd_sel, liftoffs, tkof_accel_starts, climbs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269/64.0)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860/64.0)
        self.assertEqual(node[1].value, 170)

    def test_derive__masked(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        spd_sel.array[267:272] = np.ma.masked
        node = self.node_class()
        node.derive(spd_sel, self.liftoffs, self.tkof_accel_starts, self.climbs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 269)
        self.assertEqual(node[0].value, 120)
        self.assertEqual(node[1].index, 860)
        self.assertEqual(node[1].value, 170)

    def test_derive__fully_masked(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170, 400, 170), (190, 130, 192, 192, 448)))
        # fully mask first liftoff airspeed selected
        spd_sel.array[0:353] = np.ma.masked
        node = self.node_class()
        node.derive(spd_sel, self.liftoffs, self.tkof_accel_starts, self.climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 860)
        self.assertEqual(node[0].value, 170)

    def test_derive__real(self):
        '''
        Test reflecting data we have seen where airspeed selected starts as
        previous flights landing speed before being changed before takeoff,
        this data at the start of the flight was causing landing speed to be
        used when looking back through the last 5 superframes
        '''
        liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=500),
        ])
        climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=1000),
        ])
        tkof_accel_start = KTI(name='Takeoff Acceleration Start', items=[
            KeyTimeInstance(name='Takeoff Acceleration Start', index=450),
        ])
        node = self.node_class()
        airspeed_selected = P('Airspeed Selected', np.ma.concatenate((np.ones(400) * 138, np.ones(250) * 110, np.ones(1350) * 140)))
        node.derive(airspeed_selected, liftoffs, tkof_accel_start, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 500)
        self.assertEqual(node[0].value, 110)


class TestAirspeedSelectedAtTakeoffAccelerationStart(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedSelectedAtTakeoffAccelerationStart
        self.tkof_accel_starts = KTI(name='Takeoff Acceleration Start', items=[
            KeyTimeInstance(name='Takeoff Acceleration Start', index=411),
        ])
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=438),
        ])
        self.spd_sel = P(
            ' Airspeed Selected',
            np.ma.concatenate((
                np.ma.repeat(137, 435),
                np.linspace(137, 147, 5),
                np.ma.repeat(147, 100)
            ))
        )

    def test_can_operate(self):
        self.assertEqual(self.node_class().get_operational_combinations(),
                         [('Airspeed Selected', 'Takeoff Acceleration Start', 'Liftoff')])

    def test_derive(self):
        '''
        Test values were chosen to reflect real data seen and fail if
        incorrect methods are used
        '''
        node = self.node_class()
        node.derive(self.spd_sel, self.tkof_accel_starts, self.liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 411)
        self.assertEqual(node[0].value, 137)

    def test_derive__superframe(self):
        tkof_accel_starts = KTI(name='Takeoff Acceleration Start', items=[
            KeyTimeInstance(name='Takeoff Acceleration Start', index=411/64.0),
        ])
        liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=438/64.0),
        ])
        spd_sel = P(
            'Airspeed Selected',
            np.ma.concatenate((
                np.ma.repeat(137, 435),
                np.linspace(137, 147, 5),
                np.ma.repeat(147, 100)
                ))[::64],
            frequency=1/64.0
        )
        node = self.node_class(frequency=1/64.0)
        node.derive(spd_sel, tkof_accel_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 411/64.0)
        self.assertEqual(node[0].value, 137)

    def test_derive__masked(self):
        self.spd_sel.array[405:420] = np.ma.masked
        node = self.node_class()
        node.derive(self.spd_sel, self.tkof_accel_starts, self.liftoffs, )
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 411)
        self.assertEqual(node[0].value, 137)

    def test_derive__fully_masked(self):
        self.spd_sel.array[0:500] = np.ma.masked
        node = self.node_class()
        node.derive(self.spd_sel, self.tkof_accel_starts, self.liftoffs)
        self.assertEqual(len(node), 0)


class TestAirspeedMinusV2AtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2AtLiftoff
        self.operational_combinations = [('Airspeed Minus V2', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2At35FtDuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2At35FtDuringTakeoff
        self.operational_combinations = [('Airspeed Minus V2', 'Takeoff')]

    def test_derive(self):
        spd = P('Airspeed Minus V2', np.ma.arange(0, 100, 4))
        takeoff = buildsection('Takeoff', 1, 11.75)

        node = self.node_class()
        node.derive(spd, takeoff)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 47)
        self.assertEqual(node[0].index, 11.75)


class TestAirspeedMinusV235To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV235To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2For3Sec35To1000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2For3Sec35To1000FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV235ToClimbAccelerationStartMax(unittest.TestCase):


    def setUp(self):
        self.node_class = AirspeedMinusV235ToClimbAccelerationStartMax
        self.operational_combinations = [('Airspeed Minus V2', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Airspeed Minus V2',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Airspeed Minus V2 35 To Climb Acceleration Start Max', items=[
            KeyPointValue(name='Airspeed Minus V2 35 To Climb Acceleration Start Max', index=3, value=7),
        ]))


class TestAirspeedMinusV235ToClimbAccelerationStartMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedMinusV235ToClimbAccelerationStartMin
        self.operational_combinations = [('Airspeed Minus V2', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Airspeed Minus V2',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Airspeed Minus V2 35 To Climb Acceleration Start Min', items=[
            KeyPointValue(name='Airspeed Minus V2 35 To Climb Acceleration Start Min', index=1.4, value=2.8),
        ]))


class TestAirspeedMinusV2For3Sec35ToClimbAccelerationStartMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35ToClimbAccelerationStartMax
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Airspeed Minus V2 For 3 Sec',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Max', items=[
            KeyPointValue(name='Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Max', index=3, value=7),
        ]))


class TestAirspeedMinusV2For3Sec35ToClimbAccelerationStartMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35ToClimbAccelerationStartMin
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Airspeed Minus V2 For 3 Sec',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Min', items=[
            KeyPointValue(name='Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Min', index=1.4, value=2.8),
        ]))


########################################
# Airspeed: Minus Minimum Airspeed


class TestAirspeedMinusMinimumAirspeedAbove10000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeedAbove10000FtMin
        self.operational_combinations = [('Airspeed Minus Minimum Airspeed', 'Altitude STD Smoothed')]
        self.function = min_value
        self.second_param_method_calls = [('slices_above', (10000,), {})]

    def test_derive(self):
        air_spd = P('Airspeed Minus Minimum Airspeed', np.ma.arange(200, 241))
        alt_std = P('Altitude STD Smoothed', np.ma.concatenate((np.arange(20), np.arange(20, -1, -1))) * 1000)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, alt_std)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=10, value=210),
        ]))


class TestAirspeedMinusMinimumAirspeed35To10000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeed35To10000FtMin

    def test_derive(self):
        air_spd = P('Airspeed Minus Minimum Airspeed', np.ma.arange(200, 241))
        array_start = np.concatenate((np.arange(0, 100, 20), np.arange(100, 9000, 600)))
        alt_array = np.ma.concatenate((array_start, array_start[-1:None:-1], [0]))
        alt_std = P('Altitude STD Smoothed', alt_array + 500)
        alt_aal = P('Altitude AAL For Flight Phases', alt_array)
        init_climbs = buildsection('Initial Climb', 1.75, 6.5)
        climbs = buildsection('Climb', 6.5, 20)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, alt_aal, alt_std, init_climbs, climbs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=2, value=202),
        ]))


class TestAirspeedMinusMinimumAirspeed10000To50FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeed10000To50FtMin

    def test_derive(self):
        air_spd = P('Airspeed Minus Minimum Airspeed', np.ma.arange(200, 241))
        array_start = np.concatenate((np.arange(0, 100, 20), np.arange(100, 9000, 600)))
        alt_array = np.ma.concatenate((array_start, array_start[-1:None:-1], [0]))
        alt_std = P('Altitude STD Smoothed', alt_array + 500)
        alt_aal = P('Altitude AAL For Flight Phases', alt_array)
        descents = buildsection('Descent', 21, 39)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, alt_aal, alt_std, descents)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=21, value=221),
        ]))


class TestAirspeedMinusMinimumAirspeedDuringGoAroundMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeedDuringGoAroundMin
        self.operational_combinations = [('Airspeed Minus Minimum Airspeed', 'Go Around And Climbout')]
        self.function = min_value

    def test_derive(self):
        air_spd = P('Airspeed Minus Minimum Airspeed', np.ma.arange(200, 241))
        go_around = buildsections('Go Around And Climbout', [10, 15], [35, 40])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, go_around)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=10, value=210),
            KeyPointValue(name=name, index=35, value=235),
        ]))


########################################
# Airspeed: Minus Airspeed Selected

class TestAirspeedMinusAirspeedSelectedFor3Sec1000To500FtMax(unittest.TestCase,
                                                             NodeTest):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMax
        self.operational_combinations = [
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             )]
        self.air_rel_fms = P(
            name='Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
            array=np.ma.concatenate((np.linspace(40, 0, 30), np.linspace(1, 100, 20))),
            frequency=2,
        )
        self.air_rel = P(
            name='Airspeed Minus Airspeed Selected For 3 Sec',
            array=np.ma.concatenate((np.linspace(20, -10, 30), np.linspace(-8, 9, 20))),
            frequency=2,
        )
        self.aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array(np.linspace(1100, 400, 50)),
            frequency=2,
        )
        self.duration = A('HDF Duration', value=50)

    def test_derive_no_fms(self):
        node = self.node_class()
        node.derive(self.air_rel, None, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertAlmostEqual(node[0].value, 9, places=0)

    def test_derive_fms(self):
        node = self.node_class()
        node.derive(None, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 38)
        self.assertAlmostEqual(node[0].value, 43, places=0)

    def test_derive_both(self):
        # should match test_derive_fms result
        node = self.node_class()
        node.derive(self.air_rel, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 38)
        self.assertAlmostEqual(node[0].value, 43, places=0)


class TestAirspeedMinusAirspeedSelectedFor3Sec1000To500FtMin(unittest.TestCase,
                                                             NodeTest):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMin
        self.operational_combinations = [
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             )]
        self.air_rel_fms = P(
            name='Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
            array=np.ma.concatenate((np.linspace(40, 0, 30), np.linspace(1, 100, 20))),
            frequency=2,
        )
        self.air_rel = P(
            name='Airspeed Minus Airspeed Selected For 3 Sec',
            array=np.ma.concatenate((np.linspace(20, -10, 30), np.linspace(-8, 9, 20))),
            frequency=2,
        )
        self.aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array(np.linspace(1100, 400, 50)),
            frequency=2,
        )
        self.duration = A('HDF Duration', value=50)

    def test_derive_no_fms(self):
        node = self.node_class()
        node.derive(self.air_rel, None, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, -10, places=0)

    def test_derive_fms(self):
        node = self.node_class()
        node.derive(None, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, 0, places=0)

    def test_derive_both(self):
        # should match test_derive_fms result
        node = self.node_class()
        node.derive(self.air_rel, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, 0, places=0)


class TestAirspeedMinusAirspeedSelectedFor3Sec500To20FtMax(unittest.TestCase,
                                                           NodeTest):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFor3Sec500To20FtMax
        self.operational_combinations = [
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             )]
        self.air_rel_fms = P(
            name='Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
            array=np.ma.concatenate((np.linspace(40, 0, 30), np.linspace(1, 100, 20))),
            frequency=2,
        )
        self.air_rel = P(
            name='Airspeed Minus Airspeed Selected For 3 Sec',
            array=np.ma.concatenate((np.linspace(20, -10, 30), np.linspace(-8, 9, 20))),
            frequency=2,
        )
        self.aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array(np.linspace(600, 0, 50)),
            frequency=2,
        )
        self.duration = A('HDF Duration', value=50)

    def test_derive_no_fms(self):
        node = self.node_class()
        node.derive(self.air_rel, None, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 12)
        self.assertAlmostEqual(node[0].value, 8, places=0)

    def test_derive_fms(self):
        node = self.node_class()
        node.derive(None, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 44)
        self.assertAlmostEqual(node[0].value, 74, places=0)

    def test_derive_both(self):
        # should match test_derive_fms result
        node = self.node_class()
        node.derive(self.air_rel, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 44)
        self.assertAlmostEqual(node[0].value, 74, places=0)


class TestAirspeedMinusAirspeedSelectedFor3Sec500To20FtMin(unittest.TestCase,
                                                           NodeTest):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFor3Sec500To20FtMin
        self.operational_combinations = [
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             ),
            ('Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
             'Airspeed Minus Airspeed Selected For 3 Sec',
             'Altitude AAL For Flight Phases',
             'HDF Duration'
             )]
        self.air_rel_fms = P(
            name='Airspeed Minus Airspeed Selected (FMS) For 3 Sec',
            array=np.ma.concatenate((np.linspace(40, 0, 30), np.linspace(1, 100, 20))),
            frequency=2,
        )
        self.air_rel = P(
            name='Airspeed Minus Airspeed Selected For 3 Sec',
            array=np.ma.concatenate((np.linspace(20, -10, 30), np.linspace(-8, 9, 20))),
            frequency=2,
        )
        self.aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array(np.linspace(600, 0, 50)),
            frequency=2,
        )
        self.duration = A('HDF Duration', value=50)

    def test_derive_no_fms(self):
        node = self.node_class()
        node.derive(self.air_rel, None, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, -10, places=0)

    def test_derive_fms(self):
        node = self.node_class()
        node.derive(None, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, 0, places=0)

    def test_derive_both(self):
        # should match test_derive_fms result
        node = self.node_class()
        node.derive(self.air_rel, self.air_rel_fms, self.aal, self.duration)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertAlmostEqual(node[0].value, 0, places=0)


########################################
# Airspeed: Relative


class TestAirspeedRelativeAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedRelativeAtTouchdown
        self.operational_combinations = [('Airspeed Relative', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative500To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelative20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelative20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec1000To500FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Airspeed Minus VLS

class TestAirspeedMinusVLSAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSAtTouchdown
        self.operational_combinations = [('Airspeed Minus VLS', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLS1000To500FtMax(unittest.TestCase,
                                         CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLS1000To500FtMax
        self.operational_combinations = [('Airspeed Minus VLS',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLS1000To500FtMin(unittest.TestCase,
                                         CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLS1000To500FtMin
        self.operational_combinations = [('Airspeed Minus VLS',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLS500To20FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLS500To20FtMax
        self.operational_combinations = [('Airspeed Minus VLS',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLS500To20FtMin(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLS500To20FtMin
        self.operational_combinations = [('Airspeed Minus VLS',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLSFor3Sec1000To500FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSFor3Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Minus VLS For 3 Sec',
                                          'Altitude AAL For Flight Phases',
                                          'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLSFor3Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSFor3Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Minus VLS For 3 Sec',
                                          'Altitude AAL For Flight Phases',
                                          'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLSFor3Sec500To20FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSFor3Sec500To20FtMax
        self.operational_combinations = [('Airspeed Minus VLS For 3 Sec',
                                          'Altitude AAL For Flight Phases',
                                          'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusVLSFor3Sec500To20FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSFor3Sec500To20FtMin
        self.operational_combinations = [('Airspeed Minus VLS For 3 Sec',
                                          'Altitude AAL For Flight Phases',
                                          'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')

##############################################################################
# Airspeed: Configuration


class TestAirspeedWithConfigurationMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithConfigurationMax
        self.operational_combinations = [
            ('Configuration', 'Airspeed', 'Fast'),
        ]
        self.mapping = {
            0: '0',
            1: '1',
            2: '1+F',
            3: '1*',
            4: '2',
            5: '2*',
            6: '3',
            7: '4',
            8: '5',
            9: 'Full',
        }

    def test_derive(self):
        array = np.ma.array((0, 1, 1, 2, 2, 4, 6, 4, 4, 2, 1, 0, 0, 0, 0, 0))
        conf = M(name='Configuration', array=array, values_mapping=self.mapping)
        air_spd = P(name='Airspeed', array=np.ma.arange(16))
        fast = buildsection('Fast', 0, 16)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, conf, fast)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=10, name='Airspeed With Configuration 1 Max'),
            KeyPointValue(index=9, value=9, name='Airspeed With Configuration 1+F Max'),
            KeyPointValue(index=8, value=8, name='Airspeed With Configuration 2 Max'),
            KeyPointValue(index=6, value=6, name='Airspeed With Configuration 3 Max'),
        ]))


class TestAirspeedWithConfiguration1FExcludingTransitionMaxunittest(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithConfiguration1FExcludingTransitionMax
        self.operational_combinations = [
            ('Configuration Excluding Transition', 'Airspeed',),
        ]
        self.mapping = {
            0: '0',
            1: '1',
            2: '1+F',
            3: '1*',
            4: '2',
            5: '2*',
            6: '3',
            7: '4',
            8: '5',
            9: 'Full',
        }

    def test_derive(self):
        array = np.ma.array((0, 1, 1, 2, 2, 4, 6, 4, 4, 2, 1, 0, 0, 0, 0, 0))
        conf = M(name='Configuration Excluding Transition', array=array, values_mapping=self.mapping)
        air_spd = P(name='Airspeed', array=np.ma.arange(16))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, conf)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=4, value=4.0, name='Airspeed With Configuration 1+F Excluding Transition Max'),
            KeyPointValue(index=9, value=9.0, name='Airspeed With Configuration 1+F Excluding Transition Max'),
        ]))

class TestAirspeedRelativeWithConfigurationDuringDescentMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeWithConfigurationDuringDescentMin
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [
            ('Configuration', 'Airspeed Relative', 'Descent To Flare'),
        ]
        self.mapping = {
            0: '0',
            1: '1',
            2: '1+F',
            3: '1*',
            4: '2',
            5: '2*',
            6: '3',
            7: '4',
            8: '5',
            9: 'Full',
        }

    def test_derive(self):
        array = np.ma.array((0, 1, 1, 2, 2, 4, 6, 4, 4, 2, 1, 0, 0, 0, 0, 0))
        array = np.ma.concatenate((array, array[::-1]))
        conf = M(name='Configuration', array=array, values_mapping=self.mapping)
        array = np.ma.concatenate((np.arange(16), np.arange(16, 0, -1)))
        air_spd = P(name='Airspeed', array=array)
        descent = buildsection('Descent To Flare', 16, 30)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, conf, descent)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=30, value=2, name='Airspeed Relative With Configuration 1 During Descent Min'),
            KeyPointValue(index=28, value=4, name='Airspeed Relative With Configuration 1+F During Descent Min'),
            KeyPointValue(index=26, value=6, name='Airspeed Relative With Configuration 2 During Descent Min'),
            KeyPointValue(index=25, value=7, name='Airspeed Relative With Configuration 3 During Descent Min'),
        ]))


##############################################################################
# Airspeed: Flap


class TestAirspeedWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapMax
        self.operational_combinations = [
            ('Airspeed', 'Fast', 'Flap Lever', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Fast', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Fast', 'Flap Lever'),
        ]

    def test_derive(self):
        array = np.ma.repeat((0, 5, 10), 10)
        mapping = {0: '0', 5: '5', 10: '10'}
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        flap_exc_trans = M('Flap Excluding Transition', array.copy(), values_mapping=mapping)
        air_spd = P('Airspeed', np.ma.arange(30))
        fast = buildsection('Fast', 0, 30)
        flap_inc_trans.array[19] = np.ma.masked  # mask the max value

        name = self.node_class.get_name()
        node = self.node_class()

        expected = KPV(name=name, items=[
            KeyPointValue(index=29, value=29, name='Airspeed With Flap Including Transition 10 Max'),
            KeyPointValue(index=18, value=18, name='Airspeed With Flap Including Transition 5 Max'),  # 19 was masked
            KeyPointValue(index=29, value=29, name='Airspeed With Flap Excluding Transition 10 Max'),
            KeyPointValue(index=19, value=19, name='Airspeed With Flap Excluding Transition 5 Max'),
        ])

        node.derive(air_spd, None, None, flap_inc_trans, flap_exc_trans, fast)
        self.assertEqual(sorted(node, key=lambda k: k.index or False),
                         sorted(expected, key=lambda k: k.index or False))

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (5.5, 10.1, 20.9)})
    def test_derive_fractional_settings(self):
        array = np.ma.repeat((0, 5.5, 10.1, 20.9), 5)
        mapping = {0: '0', 5.5: '5.5', 10.1: '10.1', 20.9: '20.9'}
        flap_lever = M('Flap Lever', array.copy(), values_mapping=mapping)
        flap_synth = M('Flap Lever (Synthetic)', array.copy(), values_mapping=mapping)
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        flap_exc_trans = M('Flap Excluding Transition', array.copy(), values_mapping=mapping)
        air_spd = P('Airspeed', np.ma.arange(20))
        fast = buildsection('Fast', 0, 20)

        node = self.node_class()
        node.derive(air_spd, None, None, flap_inc_trans, None, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 5.5 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 10.1 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 20.9 Max')

        node = self.node_class()
        node.derive(air_spd, None, None, None, flap_exc_trans, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap Excluding Transition 5.5 Max')

        node = self.node_class()
        node.derive(air_spd, flap_lever, None, None, None, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')

        node = self.node_class()
        node.derive(air_spd, None, flap_synth, None, None, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')

        node = self.node_class()
        node.derive(air_spd, flap_lever, flap_synth, flap_inc_trans, flap_exc_trans, fast)
        self.assertEqual(len(node), 9)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 5.5 Max')
        self.assertEqual(node[6].name, 'Airspeed With Flap Excluding Transition 5.5 Max')

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (0, 1, 5, 10, 15, 20)})
    def test_derive_Boeing(self):
        airspeed=P('Airspeed', array=np.concatenate((np.arange(20), (np.repeat([20, 23, 25, 27, 29, 31, 33, 35, 37, 39], 6)))))
        array = np.concatenate((np.zeros(3), np.ones(7), np.ones(9) * 5, np.ones(9) * 10, np.ones(11) * 15, np.ones(1) * 20))
        array = np.ma.concatenate((array, array[::-1]))
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 20: '20'}
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        fast = buildsection('Fast', 0, 80)
        array = np.array([0.0, 0.1, 0.6, 1.0] + [1.2, 1.5, 1.9, 2.5, 3.5, 4.5] +
                         [5.0, 5.2, 5.9, 6.5, 7.0, 7.9, 8.7, 9.0, 9.8, 10.0] +
                         [10.0, 10.2, 10.9, 11.5, 12.0, 12.9, 13.7, 14.0, 14.9, 15.0] +
                         [15.0, 15.2, 15.9, 16.5, 17.0, 17.9, 18.7, 19.0, 19.5, 20.0])
        array = np.concatenate((array, array[::-1]))

        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, None, fast)

        self.assertEqual(len(node), 5)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 1 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 5 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 10 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 15 Max')
        self.assertEqual(node[4].name, 'Airspeed With Flap Including Transition 20 Max')
        self.assertEqual(node[0].index, 74)
        self.assertEqual(node[1].index, 68)
        self.assertEqual(node[2].index, 56)
        self.assertEqual(node[3].index, 50)
        self.assertEqual(node[4].index, 39)
        self.assertEqual(node[0].value, 39)
        self.assertEqual(node[1].value, 37)
        self.assertEqual(node[2].value, 33)
        self.assertEqual(node[3].value, 31)
        self.assertEqual(node[4].value, 27)

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (0, 1, 5, 10, 15, 20)})
    def test_derive_Boeing_flap_angle_not_available(self):
        airspeed=P('Airspeed', array=np.concatenate((np.arange(20), (np.repeat([20, 23, 25, 27, 29, 31, 33, 35, 37, 39], 6)))))
        array = np.concatenate((np.zeros(3), np.ones(7), np.ones(9) * 5, np.ones(9) * 10, np.ones(11) * 15, np.ones(1) * 20))
        array = np.ma.concatenate((array, array[::-1]))
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 20: '20'}
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        fast = buildsection('Fast', 0, 80)

        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, None, fast)

        self.assertEqual(len(node), 5)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 1 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 5 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 10 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 15 Max')
        self.assertEqual(node[4].name, 'Airspeed With Flap Including Transition 20 Max')
        self.assertEqual(node[0].index, 74)
        self.assertEqual(node[1].index, 68)
        self.assertEqual(node[2].index, 56)
        self.assertEqual(node[3].index, 50)
        self.assertEqual(node[4].index, 39)
        self.assertEqual(node[0].value, 39)
        self.assertEqual(node[1].value, 37)
        self.assertEqual(node[2].value, 33)
        self.assertEqual(node[3].value, 31)
        self.assertEqual(node[4].value, 27)

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (0, 1, 5, 10, 15, 20)})
    def test_derive_Boeing_flap_including_transition_not_available(self):
        airspeed=P('Airspeed', array=np.concatenate((np.arange(20), (np.repeat([20, 23, 25, 27, 29, 31, 33, 35, 37, 39], 6)))))
        array = np.concatenate((np.zeros(3), np.ones(7), np.ones(9) * 5, np.ones(9) * 10, np.ones(11) * 15, np.ones(1) * 20))
        array = np.ma.concatenate((array, array[::-1]))
        fast = buildsection('Fast', 0, 80)
        array = np.array([0.0, 0.1, 0.6, 1.0] + [1.2, 1.5, 1.9, 2.5, 3.5, 4.5,
                                                 5.0, 5.2, 5.9, 6.5, 7.0, 7.9, 8.7, 9.0, 9.8, 10.0,
                          10.0, 10.2, 10.9, 11.5, 12.0, 12.9, 13.7, 14.0, 14.9, 15.0,
                          15.0, 15.2, 15.9, 16.5, 17.0, 17.9, 18.7, 19.0, 19.5, 20.0])
        array = np.ma.concatenate((array, array[::-1]))

        node = self.node_class()
        node.derive(airspeed, None, None, None, None, fast)

        self.assertEqual(len(node), 0)

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (0, 1, 5, 10, 15, 20)})
    def test_derive_Boeing_masked_data(self):
        airspeed=P('Airspeed', array=np.concatenate((np.arange(20), (np.repeat([20, 23, 25, 27, 29, 31, 33, 35, 37, 39], 6)))))
        array = np.concatenate((np.zeros(3), np.ones(7), np.ones(9) * 5, np.ones(9) * 10, np.ones(11) * 15, np.ones(1) * 20))
        array = np.ma.concatenate((array, array[::-1]))
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 20: '20'}
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        fast = buildsection('Fast', 0, 80)
        array = np.array([0.0, 0.1, 0.6, 1.0] + [1.2, 1.5, 1.9, 2.5, 3.5, 4.5] +
                         [5.0, 5.2, 5.9, 6.5, 7.0, 7.9, 8.7, 9.0, 9.8, 10.0] +
                             [10.0, 10.2, 10.9, 11.5, 12.0, 12.9, 13.7, 14.0, 14.9, 15.0] +
                             [15.0, 15.2, 15.9, 16.5, 17.0, 17.9, 18.7, 19.0, 19.5, 20.0])
        array = np.concatenate((array, array[::-1]))
        flap_angle=P('Flap Angle', array)
        flap_angle.array.mask = True

        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, None, fast)

        self.assertEqual(len(node), 5)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 1 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 5 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 10 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 15 Max')
        #self.assertEqual(node[4].name, 'Airspeed With Flap Including Transition 20 Max')
        self.assertEqual(node[0].index, 74)
        self.assertEqual(node[1].index, 68)
        self.assertEqual(node[2].index, 56)
        self.assertEqual(node[3].index, 50)
        #self.assertEqual(node[4].index, 39)
        self.assertEqual(node[0].value, 39)
        self.assertEqual(node[1].value, 37)
        self.assertEqual(node[2].value, 33)
        self.assertEqual(node[3].value, 31)
        #self.assertEqual(node[4].value, 27)

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES', {'flap': (0, 1, 5, 10, 15, 20)})
    def test_derive_Boeing_fast_flap_angle_rate_of_change(self):
        airspeed=P('Airspeed', array=np.concatenate((np.arange(20), (np.repeat([20, 23, 25, 27, 29, 31, 33, 35, 37, 39], 6)))))
        array = np.concatenate((np.zeros(3), np.ones(7), np.ones(9) * 5, np.ones(9) * 10, np.ones(11) * 15, np.ones(1) * 20))
        array = np.ma.concatenate((array, array[::-1]))
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 20: '20'}
        flap_inc_trans = M('Flap Including Transition', array.copy(), values_mapping=mapping)
        fast = buildsection('Fast', 0, 80)
        angle_array = np.array([0.0, 0.1, 0.6, 1.0] + [1.9, 2.3, 2.5, 3.5, 4.5] +
                               [4.9, 5.0, 5.0, 7.5, 7.6, 7.9, 8.5, 8.9, 9.5, 9.8] +
                               [10.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1, 15.1] +
                               [15.3, 15.5, 15.9, 16.5, 17.0, 17.9, 18.7, 19.0, 19.5, 20.0])
        angle_array = np.concatenate((angle_array, angle_array[::-1]))

        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, None, fast)

        self.assertEqual(len(node), 5)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 1 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 5 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 10 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 15 Max')
        self.assertEqual(node[4].name, 'Airspeed With Flap Including Transition 20 Max')
        self.assertEqual(node[0].index, 74)
        self.assertEqual(node[1].index, 68)
        self.assertEqual(node[2].index, 56)
        self.assertEqual(node[3].index, 50)
        self.assertEqual(node[4].index, 39)
        self.assertEqual(node[0].value, 39)
        self.assertEqual(node[1].value, 37)
        self.assertEqual(node[2].value, 33)
        self.assertEqual(node[3].value, 31)
        self.assertEqual(node[4].value, 27)


class TestAirspeedWithFlapMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapMin
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Airspeed', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airspeed', 'Airborne'),
        ]

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestAirspeedWithFlapAndSlatExtendedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapAndSlatExtendedMax
        self.operational_combinations = [
            ('Flap Excluding Transition', 'Slat Excluding Transition', 'Airspeed', 'Fast'),
            ('Flap Including Transition', 'Slat Including Transition', 'Airspeed', 'Fast'),
        ]

    def test_derive_basic(self):
        array = np.ma.array((0, 0, 5, 10, 10, 10, 15, 15, 15, 35) * 2)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_inc_trsn = M('Flap Including Transition', array, values_mapping=mapping)
        array = np.ma.array((0, 0, 5, 10, 15, 35, 35, 15, 10, 0) * 2)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_exc_trsn = M('Flap Excluding Transition', array, values_mapping=mapping)

        array = np.ma.array((0, 10, 10, 10, 10, 10, 20, 20, 20, 20) * 2)
        mapping = {int(s): str(s) for s in np.ma.unique(array)}
        slat_inc_trsn = M('Slat Including Transition', array, values_mapping=mapping)
        array = np.ma.array((0, 10, 10, 10, 20, 20, 20, 20, 10, 10) * 2)
        mapping = {int(s): str(s) for s in np.ma.unique(array)}
        slat_exc_trsn = M('Slat Excluding Transition', array, values_mapping=mapping)

        airspeed = P('Airspeed', np.ma.arange(0, 200, 10))
        airspeed.array[1] = 500.0  # excluded for inc - outside fast section.
        airspeed.array[9] = 500.0  # selected for exc - max value.
        fast = buildsection('Fast', 5, None)

        node = self.node_class()
        node.derive(airspeed, flap_exc_trsn, flap_inc_trsn, slat_exc_trsn, slat_inc_trsn, fast)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=9.0, value=500.0, name='Airspeed With Flap Excluding Transition 0 And Slat Extended Max'),
            KeyPointValue(index=11.0, value=110.0, name='Airspeed With Flap Including Transition 0 And Slat Extended Max'),
        ])


class TestAirspeedWithFlapIncludingTransition20AndSlatFullyExtendedMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedWithFlapIncludingTransition20AndSlatFullyExtendedMax

    def test_can_operate(self):
        req_params = ('Flap Including Transition', 'Slat Including Transition', 'Airspeed', 'Fast')
        family = A('Family', value='B767')
        self.assertFalse(self.node_class.can_operate(req_params, family=family), msg='KPV should not work for B767')
        family = A('Family', value='B777')
        self.assertTrue(self.node_class.can_operate(req_params, family=family))

    @patch('analysis_engine.key_point_values.at.get_slat_map')
    def test_derive_basic(self, get_slat_map):
        get_slat_map.return_value = {x: str(x) for x in (0, 22, 32)}
        b777 = A('Family', value='B777')
        flap_values_mapping = {0: '0', 20: '20', 5: '5', 30: '30', 15: '15'}
        slat_values_mapping = {0: '0', 32: '32', 22: '22'}

        flap_inc_array = np.ma.concatenate((np.zeros(5), np.ones(5) * 5, np.ones(10) * 20, np.ones(5) * 30))
        flap_inc_trsn = M('Flap Including Transition', flap_inc_array, values_mapping=flap_values_mapping)

        slat_inc_array = np.ma.concatenate((np.zeros(2), np.ones(13) * 22, np.ones(10) * 32))
        slat_inc_trsn = M('Slat Including Transition', slat_inc_array, values_mapping=slat_values_mapping)

        airspeed = P('Airspeed', np.ma.arange(300, 200, -4))
        fast = buildsection('Fast', 5, None)

        node = self.node_class()
        node.derive(airspeed, flap_inc_trsn, slat_inc_trsn, fast, b777)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=15.0, value=240.0, name='Airspeed With Flap Including Transition 20 And Slat Fully Extended Max'),
        ])


class TestAirspeedWithFlapDuringClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringClimbMax
        self.operational_combinations = [
            ('Airspeed', 'Climb', 'Flap Lever', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Climb', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Climb', 'Flap Lever'),
        ]

    def test_derive_basic(self):
        array = np.ma.array((0, 0, 5, 10, 10, 10, 15, 15, 15, 30))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_inc_trans = M('Flap Including Transition', array, values_mapping=mapping)
        array = np.ma.array((0, 0, 5, 10, 15, 30, 30, 15, 10, 0))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_exc_trans = M('Flap Excluding Transition', array, values_mapping=mapping)
        airspeed = P('Airspeed', np.ma.arange(0, 100, 10))
        climb = buildsection('Climbing', 2, 7)
        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, flap_exc_trans, climb)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=2.0, value=20.0, name='Airspeed With Flap Including Transition 5 During Climb Max'),
            KeyPointValue(index=2.0, value=20.0, name='Airspeed With Flap Excluding Transition 5 During Climb Max'),
            KeyPointValue(index=5.0, value=50.0, name='Airspeed With Flap Including Transition 10 During Climb Max'),
            KeyPointValue(index=6.0, value=60.0, name='Airspeed With Flap Excluding Transition 30 During Climb Max'),
            KeyPointValue(index=7.0, value=70.0, name='Airspeed With Flap Excluding Transition 15 During Climb Max'),
            KeyPointValue(index=8.0, value=80.0, name='Airspeed With Flap Including Transition 15 During Climb Max'),
            KeyPointValue(index=8.0, value=80.0, name='Airspeed With Flap Excluding Transition 10 During Climb Max'),
        ])


class TestAirspeedWithFlapDuringClimbMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringClimbMin
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed', 'Climb'),
            ('Flap Lever (Synthetic)', 'Airspeed', 'Climb'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airspeed', 'Climb'),
        ]

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestAirspeedWithFlapDuringDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringDescentMax
        self.operational_combinations = [
            ('Airspeed', 'Descent', 'Flap Lever', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Descent', 'Flap Lever (Synthetic)', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Descent', 'Flap Lever'),
        ]

    def test_derive_basic(self):
        array = np.ma.array((0, 0, 5, 10, 10, 10, 15, 15, 15, 30))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_inc_trans = M('Flap Including Transition', array, values_mapping=mapping)
        array = np.ma.array((0, 0, 5, 10, 15, 30, 30, 15, 10, 0))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_exc_trans = M('Flap Excluding Transition', array, values_mapping=mapping)
        airspeed = P('Airspeed', np.ma.arange(100, 0, -10))
        desc = buildsection('Descending', 2, 7)
        node = self.node_class()
        node.derive(airspeed, None, None, flap_inc_trans, flap_exc_trans, desc)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=2.0, value=80.0, name='Airspeed With Flap Including Transition 5 During Descent Max'),
            KeyPointValue(index=2.0, value=80.0, name='Airspeed With Flap Excluding Transition 5 During Descent Max'),
            KeyPointValue(index=3.0, value=70.0, name='Airspeed With Flap Including Transition 10 During Descent Max'),
            KeyPointValue(index=3.0, value=70.0, name='Airspeed With Flap Excluding Transition 10 During Descent Max'),
            KeyPointValue(index=4.0, value=60.0, name='Airspeed With Flap Excluding Transition 15 During Descent Max'),
            KeyPointValue(index=5.0, value=50.0, name='Airspeed With Flap Excluding Transition 30 During Descent Max'),
            KeyPointValue(index=6.0, value=40.0, name='Airspeed With Flap Including Transition 15 During Descent Max'),])


class TestAirspeedWithFlapDuringDescentMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringDescentMin
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed', 'Descent To Flare'),
            ('Flap Lever (Synthetic)', 'Airspeed', 'Descent To Flare'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airspeed', 'Descent To Flare'),
        ]

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestAirspeedMinusFlapManoeuvreSpeedWithFlapDuringDescentMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusFlapManoeuvreSpeedWithFlapDuringDescentMin
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed Minus Flap Manoeuvre Speed', 'Descent To Flare'),
            ('Flap Lever (Synthetic)', 'Airspeed Minus Flap Manoeuvre Speed', 'Descent To Flare'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airspeed Minus Flap Manoeuvre Speed', 'Descent To Flare'),
        ]

    def test_derive(self):
        array = np.ma.array((0, 0, 5, 10, 10, 10, 15, 15, 15, 35))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M('Flap Lever', array, values_mapping=mapping)
        airspeed = P('Airspeed', np.ma.arange(100, 0, -10))
        descents = buildsection('Descent To Flare', 2, 7)
        node = self.node_class()
        node.derive(airspeed, flap, None, descents)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=2, value=80, name='Airspeed Minus Flap Manoeuvre Speed With Flap 5 During Descent Min'),
            KeyPointValue(index=5, value=50, name='Airspeed Minus Flap Manoeuvre Speed With Flap 10 During Descent Min'),
            KeyPointValue(index=8, value=20, name='Airspeed Minus Flap Manoeuvre Speed With Flap 15 During Descent Min'),
        ])


class TestAirspeedSelectedFMCMinusFlapManoeuvreSpeed1000to5000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedSelectedFMCMinusFlapManoeuvreSpeed1000to5000FtMin

    def test_can_operate(self):
        expected = [('Airspeed Selected (FMC)', 'Flap Manoeuvre Speed', 'Altitude AAL For Flight Phases', 'Climb')]
        self.assertEqual(self.node_class().get_operational_combinations(),
                         expected)

    def test_derive(self):
        spd_sel = P('Airspeed Selected (FMC)', np.ma.repeat((150, 250), 20))
        manoeuvre_spd = P('Flap Manoeuvre Speed', np.ma.repeat((155, 175, 200), (11, 4, 25)))
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(0, 6000, 150))
        climbs = buildsection('Climb', 6, 33)
        node = self.node_class()

        node.derive(spd_sel, manoeuvre_spd, alt_aal, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, -50)
        self.assertEqual(node[0].index, 15)


########################################
# Airspeed: Landing Gear


class TestAirspeedWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithGearDownMax
        self.operational_combinations = [('Airspeed', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        air_spd = P(
            name='Airspeed',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(air_spd, gear, airs)
        self.assertEqual(len(node), 1)
        # Only maximum in flight is taken
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 5.0)
        self.assertEqual(node[0].name, 'Airspeed With Gear Down Max')


class TestAirspeedWhileGearRetractingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedWhileGearRetractingMax
        self.operational_combinations = [('Airspeed', 'Gear Retracting')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedWhileGearExtendingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedWhileGearExtendingMax
        self.operational_combinations = [('Airspeed', 'Gear Extending')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAtGearUpSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtGearUpSelection
        self.operational_combinations = [('Airspeed', 'Gear Up Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAtGearDownSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtGearDownSelection
        self.operational_combinations = [('Airspeed', 'Gear Down Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Thrust Reversers


class TestAirspeedWithThrustReversersDeployedMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithThrustReversersDeployedMin
        self.operational_combinations = [
            ('Airspeed True', 'Thrust Reversers', 'Eng (*) EPR Max', 'Eng (*) N1 Max', 'Landing'),
            ('Airspeed True', 'Thrust Reversers', 'Eng (*) N1 Max', 'Landing'),
            ('Airspeed True', 'Thrust Reversers', 'Eng (*) EPR Max', 'Landing')]

    def test_derive_basic(self):
        air_spd=P('Airspeed True', array = np.ma.arange(100,0,-10))
        tr=M('Thrust Reversers', array=np.ma.array([0]*3+[1]+[2]*4+[1,0]),
             values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        n1=P('Eng (*) N1 Max', array=np.ma.array([40]*5+[70]*5))
        landings=buildsection('Landing', 2, 9)
        node = AirspeedWithThrustReversersDeployedMin()
        node.derive(air_spd, tr, None, n1, landings)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=7, value=30.0, name='Airspeed With Thrust Reversers Deployed Min'))

    def test_derive_with_epr(self):
        air_spd = P('Airspeed True', array = np.ma.arange(100,0,-10))
        tr = M('Thrust Reversers', array=np.ma.array([0]*3+[1]+[2]*4+[1,0]),
               values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        epr = P('Eng (*) EPR Max', array=np.ma.array([1.0]*5+[1.26]*3+[1.0]*2))
        landings=buildsection('Landing', 2, 9)
        node = AirspeedWithThrustReversersDeployedMin()
        node.derive(air_spd, tr, epr, None, landings)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=7, value=30.0, name='Airspeed With Thrust Reversers Deployed Min'))

    def test_derive_inadequate_power(self):
        air_spd=P('Airspeed True',array = np.ma.arange(100,0,-10))
        tr=M('Thrust Reversers', array=np.ma.array([0]*3+[1]+[2]*4+[1,0]),
             values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        n1=P('Eng (*) N1 Max', array=np.ma.array([40]*10))
        landings=buildsection('Landing', 2, 9)
        node = AirspeedWithThrustReversersDeployedMin()
        node.derive(air_spd, tr, None, n1, landings)
        self.assertEqual(len(node), 0)

    def test_derive_not_deployed(self):
        air_spd=P('Airspeed True',array = np.ma.arange(100,0,-10))
        tr=M('Thrust Reversers', array=np.ma.array([0]*3+[1]*6+[0]),
             values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        n1=P('Eng (*) N1 Max', array=np.ma.array([40]*5+[70]*5))
        landings=buildsection('Landing', 2, 9)
        node = AirspeedWithThrustReversersDeployedMin()
        node.derive(air_spd, tr, None, n1, landings)
        self.assertEqual(len(node), 0)


class TestAirspeedAtThrustReversersSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAtThrustReversersSelection
        self.operational_combinations = [('Airspeed', 'Thrust Reversers', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedWithThrustReversersDeployedAnyPowerMin(unittest.TestCase, NodeTest):

    def test_derive_basic(self):
        gnd_spd = P('Groundspeed True', array=np.ma.arange(100, 0, -10))
        tr = M('Thrust Reversers', array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
               values_mapping={0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})

        node = GroundspeedWithThrustReversersDeployedAnyPowerMin()
        node.derive(gnd_spd, tr)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=7, value=30.0, name='Groundspeed With Thrust Reversers Deployed Any Power Min'))


########################################
# Airspeed: Other


class TestAirspeedVacatingRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedVacatingRunway
        self.operational_combinations = [('Airspeed True', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedDuringRejectedTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringRejectedTakeoffMax
        self.operational_combinations = [('Airspeed', 'Rejected Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedBelow10000FtDuringDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedBelow10000FtDuringDescentMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTopOfDescentTo10000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedTopOfDescentTo10000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTopOfDescentTo4000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedTopOfDescentTo4000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTopOfDescentTo4000FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedTopOfDescentTo4000FtMin
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeed3000FtToTopOfClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Airspeed3000FtToTopOfClimbMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Top Of Climb')]

    def test_derive_basic(self):
        alt_aal_array = np.ma.arange(0, 20000, 100)
        alt_aal = P('Altitude AAL For Flight Phases', alt_aal_array)
        air_spd_array = np.ma.arange(100, 300)
        air_spd = P('Airspeed', air_spd_array)
        tocs = KTI('Top Of Climb', items=[KeyTimeInstance(150, 'Top Of Climb')])
        node = Airspeed3000FtToTopOfClimbMax()
        node.derive(air_spd, alt_aal, tocs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(149, 249, 'Airspeed 3000 Ft To Top Of Climb Max'))


class TestAirspeed3000FtToTopOfClimbMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Airspeed3000FtToTopOfClimbMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Top Of Climb')]

    def test_derive_basic(self):
        alt_aal_array = np.ma.arange(0, 20000, 100)
        alt_aal = P('Altitude AAL For Flight Phases', alt_aal_array)
        air_spd_array = np.ma.arange(100, 300)
        air_spd = P('Airspeed', air_spd_array)
        tocs = KTI('Top Of Climb', items=[KeyTimeInstance(150, 'Top Of Climb')])
        node = self.node_class()
        node.derive(air_spd, alt_aal, tocs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(30, 130, 'Airspeed 3000 Ft To Top Of Climb Min'))


class TestAirspeedDuringLevelFlightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedDuringLevelFlightMax
        self.operational_combinations = [('Airspeed', 'Level Flight')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithSpeedbrakeDeployedMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AirspeedWithSpeedbrakeDeployedMax
        self.operational_combinations = [('Airspeed', 'Speedbrake')]

    def test_derive_basic(self):
        air_spd = P(
            name='Airspeed',
            array=np.ma.arange(10),
        )
        spoiler_values_mapping = {0: '0', 6: '6'}
        spoiler = M(
            name='Speedbrake',
            array=np.ma.array([0, 0, 0, 0, 6, 6, 0, 0, 6, 0]),
            values_mapping=spoiler_values_mapping
        )

        node = self.node_class()
        node.derive(air_spd, spoiler)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8)
        self.assertEqual(node[0].value, 8.0)
        self.assertEqual(node[0].name, 'Airspeed With Speedbrake Deployed Max')


    def test_derive_below_threshold(self):
        air_spd = P(
            name='Airspeed',
            array=np.ma.arange(10),
        )
        spoiler_values_mapping = {0: '0', 4:'4', 6: '6'}
        spoiler = M(
            name='Speedbrake',
            array=np.ma.array([0, 0, 0, 0, 6, 6, 0, 0, 4, 0]),
            values_mapping=spoiler_values_mapping
        )

        node = self.node_class()
        node.derive(air_spd, spoiler)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 5.0)
        self.assertEqual(node[0].name, 'Airspeed With Speedbrake Deployed Max')



##############################################################################
# Airspeed: Cruise
class TestAirspeedAboveFL200Max(unittest.TestCase):
    def can_operate(self):
        opts = AirspeedAboveFL200Max.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed', 'Altitude STD Smoothed')])

    def test_derive(self):
        airspeed = P('Airspeed', array=[238, 240, 236, 234, 236, 237, 235, 239, 240])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = AirspeedAboveFL200Max()
        node.derive(airspeed, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 237)

class TestAirspeedAboveFL200Min(unittest.TestCase):
    def can_operate(self):
        opts = AirspeedAboveFL200Min.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed', 'Altitude STD Smoothed')])

    def test_derive(self):
        airspeed = P('Airspeed', array=[238, 240, 236, 234, 236, 237, 235, 239, 240])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = AirspeedAboveFL200Min()
        node.derive(airspeed, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 236)


##############################################################################
# Angle of Attack


class TestAOAWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOAWithFlapMax
        self.operational_combinations = [
            ('Flap Lever', 'AOA', 'Airborne'),
            ('Flap Lever (Synthetic)', 'AOA', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'AOA', 'Airborne'),
        ]

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestAOAWithFlapDuringClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOAWithFlapDuringClimbMax
        self.operational_combinations = [
            ('Flap Lever', 'AOA', 'Climbing'),
            ('Flap Lever (Synthetic)', 'AOA', 'Climbing'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'AOA', 'Climbing'),
        ]

    def test_derive_basic(self):
        aoa = P('AOA', array=np.arange(30))

        flap_values_mapping = {0: '0', 10: '10'}
        flap_array = np.ma.concatenate((np.ones(10) * 10, np.zeros(10), np.ones(10) * 10))
        flap = M('Flap Lever', array=flap_array, values_mapping=flap_values_mapping)

        flap_synth_values_mapping = {0: 'Lever 0', 1: 'Lever 1'}
        flap_synth_array = np.ma.concatenate((np.zeros(10), np.ones(10), np.zeros(10)))
        flap_synth = M('Flap Lever (Synthetic)', array=flap_synth_array, values_mapping=flap_synth_values_mapping)

        climbs = buildsections('Climbing', (15, 25))

        node = self.node_class()
        node.derive(aoa, flap, None, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 25)
        self.assertEqual(node[0].value, 25)

        node = self.node_class()
        node.derive(aoa, None, flap_synth, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 19)
        self.assertEqual(node[0].value, 19)


class TestAOAWithFlapDuringDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOAWithFlapDuringDescentMax
        self.operational_combinations = [
            ('Flap Lever', 'AOA', 'Descending'),
            ('Flap Lever (Synthetic)', 'AOA', 'Descending'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'AOA', 'Descending'),
        ]

    def test_derive_basic(self):
        aoa = P('AOA', array=np.arange(30, 0, -1))

        flap_values_mapping = {0: '0', 10: '10'}
        flap_array = np.ma.concatenate((np.ones(10) * 10, np.zeros(10), np.ones(10) * 10))
        flap = M('Flap Lever', array=flap_array, values_mapping=flap_values_mapping)

        flap_synth_values_mapping = {0: 'Lever 0', 1: 'Lever 1'}
        flap_synth_array = np.ma.concatenate((np.zeros(10), np.ones(10), np.zeros(10)))
        flap_synth = M('Flap Lever (Synthetic)', array=flap_synth_array, values_mapping=flap_synth_values_mapping)

        climbs = buildsections('Descending', (15, 25))

        node = self.node_class()
        node.derive(aoa, flap, None, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 20)
        self.assertEqual(node[0].value, 10)

        node = self.node_class()
        node.derive(aoa, None, flap_synth, climbs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 15)


class TestAOADuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AOADuringGoAroundMax
        self.operational_combinations = [('AOA', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


##############################################################################
# MCAS

class TestAOADifference5SecMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOADifference5SecMax
        self.operational_combinations = [('AOA (L)', 'AOA (R)', 'Airborne')]
        self.function = max_value

    def test_derive(self):
        aoa_r_arr = np.sin(np.arange(1,20))
        aoa_l_arr = np.cos(np.arange(1,20))
        aoa_l = P('AOA (L)', array=aoa_l_arr)
        aoa_r = P('AOA (R)', array=aoa_r_arr)
        airs = buildsection('Airborne', 3, 15)
        node = self.node_class()
        node.derive(aoa_l, aoa_r, airs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 0.30, places=2)
        self.assertEqual(node[0].index, 3)


class TestAOAAbnormalOperationDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = AOAAbnormalOperationDuration

    def test_derive(self):
        aoa_mapping = {
            0: '-',
            1: 'AOA (L) Fail'
        }
        aoa = M('AOA Abnormal Operation', array=np.ma.array([0]*15 + [1]*3 + [0]*2), values_mapping=aoa_mapping)
        airborne = buildsection('Airborne', 16, 20)
        node=self.node_class()
        node.derive(aoa, airborne)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 2.0, places=1)
        self.assertEqual(node[0].index, 16)


class TestAOABelowStickShakerAOAMin(unittest.TestCase):
    def setUp(self):
        self.node_class = AOABelowStickShakerAOAMin

    def test_derive(self):
        x = np.linspace(0, 10, 100)
        aoa_stick_shaker = P('Stick Shaker AOA', -x*np.sin(x))
        aoa_array = -x*np.sin(x)
        aoa_array[70:90] += 2
        aoa_l = P('AOA (L)', aoa_array -1)
        aoa_r = P('AOA (R)', -x*np.sin(x))
        airborne = buildsection('Airborne', 5, 100)
        node = self.node_class()
        node.derive(aoa_stick_shaker, aoa_l, aoa_r, airborne)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, -1.0, places=1)
        self.assertEqual(node[0].index, 70)


class TestAOAStickShakerMinusAOAFlapsUpAPOffMin(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AOAStickShakerMinusAOAFlapsUpAPOffMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(family=A('Family', 'B737 MAX')),
                         [('Stick Shaker AOA', 'AOA (L)', 'AOA (R)', 'Airborne', 'Flap Including Transition',
                           'AP Engaged')])

    def test_derive(self):
        x = np.linspace(0, 10, 100)
        aoa_stick_shaker = P('Stick Shaker AOA', -x*np.sin(x))
        aoa_array = -x*np.sin(x)
        aoa_array[70:90] += 2
        aoa_l = P('AOA (L)', aoa_array -1)
        aoa_r = P('AOA (R)', -x*np.sin(x))
        airborne = buildsection('Airborne', 5, 100)
        flap = P('Flap Including Transition', array=([5]*50 + [0]*50))
        ap_values = {
            0: '-',
            1: 'Engaged',
        }
        ap = M(name='AP Engaged', array=np.ma.array([1]*50 + [0]*50, mask=[False]*100), values_mapping=ap_values)
        node = self.node_class()
        node.derive(aoa_stick_shaker, aoa_l, aoa_r, airborne, flap, ap)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, -1.0, places=1)
        self.assertEqual(node[0].index, 70)


class TestTrimDownWhileControlColumnUpDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TrimDownWhileControlColumnUpDuration

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(family=A('Family', 'B737 MAX')),
                         [('Control Column', 'AP Trim Down', 'Airborne')])

    def test_derive(self):
        pitch_mapping = {
            0: '-',
            1: 'Trim',
        }
        trim = M('AP Trim Down', np.ma.array([0]*15 + [1]*5 + [0]*20 + [1]*10), values_mapping=pitch_mapping)
        cc = P('Control Column', np.ma.arange(-25,25,1))
        airs = buildsection('Airborne', 5, 50)
        node = self.node_class()
        node.derive(cc, trim, airs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 10.0, places=1)
        self.assertEqual(node[0].index, 40)


class TestAirspeedDifference5SecMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AirspeedDifference5SecMax
        self.operational_combinations = [('Airspeed', 'Airspeed (2)', 'Airborne')]
        self.function = max_value

    def test_derive(self):
        ias_1_arr = np.sin(np.arange(1,20))
        ias_2_arr = np.cos(np.arange(1,20))
        ias_1 = P('Airspeed', array=ias_1_arr)
        ias_2 = P('Airspeed (2)', array=ias_2_arr)
        airs = buildsection('Airborne', 3, 15)
        node = self.node_class()
        node.derive(ias_1, ias_2, airs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, -0.30, places=2)
        self.assertEqual(node[0].index, 3)


class TestAOAFlapsUpAPOffMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AOAFlapsUpAPOffMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(family=A('Family', 'B737 MAX')),
                         [('AOA (L)', 'AOA (R)', 'Flap Including Transition', 'AP Engaged', 'Airborne')])

    def test_derive(self):
        aoa_l_arr = np.sin(np.arange(1,20))
        aoa_r_arr = np.cos(np.arange(1,20))
        aoa_l = P('AOA (L)', array=aoa_l_arr)
        aoa_r = P('AOA (R)', array=aoa_r_arr)
        flap = P('Flap Including Transition', array=([5]*5 + [0]*15))
        ap_values = {
            0: '-',
            1: 'Engaged',
        }
        ap = M(name='AP Engaged', array=np.ma.array([0]*10 + [1]*10, mask=[False]*20), values_mapping=ap_values)
        airs = buildsection('Airborne', 3, 20)
        climb = buildsection('Climbing', 3, 15)
        node = self.node_class()
        node.derive(aoa_l, aoa_r, flap, ap, airs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 0.99, places=2)
        self.assertEqual(node[0].index, 7)


class TestAOAAbnormalOperationDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = AOAAbnormalOperationDuration

    def test_derive(self):
        aoa_mapping = {
            0: '-',
            1: 'AOA (L) Fail'
        }
        aoa = M('AOA State', array=np.ma.array([0]*15 + [1]*3 + [0]*2), values_mapping=aoa_mapping)
        airborne = buildsection('Airborne', 16, 20)
        node=self.node_class()
        node.derive(aoa, airborne)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 2.0, places=1)
        self.assertEqual(node[0].index, 16)


class TestAOABelowStickShakerAOAMin(unittest.TestCase):
    def setUp(self):
        self.node_class = AOABelowStickShakerAOAMin

    def test_derive(self):
        x = np.linspace(0, 10, 100)
        aoa_stick_shaker = P('AOA Stick Shaker', -x*np.sin(x))
        aoa_array = -x*np.sin(x)
        aoa_array[70:90] += 2
        aoa_l = P('AOA (L)', aoa_array -1)
        aoa_r = P('AOA (R)', -x*np.sin(x))
        airborne = buildsection('Airborne', 5, 100)
        node = self.node_class()
        node.derive(aoa_stick_shaker, aoa_l, aoa_r, airborne)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, -1.0, places=1)
        self.assertEqual(node[0].index, 70)


class TestTrimDownWhileControlColumnUpDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = TrimDownWhileControlColumnUpDuration

    def test_derive(self):
        pitch_mapping = {
            0: '-',
            1: 'Trim',
        }
        trim = M('AP Trim Down', np.ma.array([0]*15 + [1]*5 + [0]*20 + [1]*10), values_mapping=pitch_mapping)
        cc = P('Control Column', np.ma.arange(-25,25,1))
        airs = buildsection('Airborne', 5, 50)
        node = self.node_class()
        node.derive(cc, trim, airs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 10.0, places=1)
        self.assertEqual(node[0].index, 40)


##############################################################################
# Autopilot


class TestAPDisengagedDuringCruiseDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APDisengagedDuringCruiseDuration
        self.operational_combinations = [('AP Engaged', 'Cruise')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

class TestATEngagedAPDisengagedOutsideClimbDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ATEngagedAPDisengagedOutsideClimbDuration
        self.operational_combinations = [('AT Engaged', 'AP Engaged', 'Vertical Speed For Flight Phases', 'Airborne', 'Takeoff')]
        self.can_operate_kwargs = {'ac_family': A('Family', value='B737 NG')}

    def test_derive(self):
        at_engaged = M('AT Engaged', array=np.ma.ones(40), values_mapping={0: '-', 1: 'Engaged'})
        ap_engaged = M('AP Engaged', array=np.ma.concatenate((np.ones(5), np.zeros(30), np.ones(5))), values_mapping={0: '-', 1: 'Engaged'})
        airs = buildsection('Airborne', 1, 39)
        vert = P('Vertical Speed For Flight Phases', array=np.ma.concatenate((np.ones(10) * 350, np.ones(30) * 250)))
        #climbs = buildsection('Climbing', 1, 10)

        node = self.node_class()
        node.derive(at_engaged, ap_engaged, vert, airs)

        name = 'AT Engaged AP Disengaged Outside Climb Duration'
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=10, value=25),
        ])
        self.assertEqual(node, expected)

    def test_derive__invalid_at_liftoff(self):
        '''
        test to check around liftoff when climb starts the sample after airborne.
        '''
        at_engaged = M('AT Engaged', array=np.ma.ones(40), values_mapping={0: '-', 1: 'Engaged'})
        ap_engaged = M('AP Engaged', array=np.ma.concatenate((np.ones(5), np.zeros(30), np.ones(5))), values_mapping={0: '-', 1: 'Engaged'})
        airs = buildsection('Airborne', 5, 39)
        vert = P('Vertical Speed For Flight Phases', array=np.ma.concatenate((np.ones(5) * 250, np.ones(5) * 350, np.ones(30) * 250)))
        #climbs = buildsection('Climbing', 6, 10)
        toff = buildsection('Takeoff', 0, 7)

        node = self.node_class()
        node.derive(at_engaged, ap_engaged, vert, airs, toff)

        name = 'AT Engaged AP Disengaged Outside Climb Duration'
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=10, value=25),
        ])
        self.assertEqual(node, expected)


class TestATDisengagedAPEngagedDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ATDisengagedAPEngagedDuration
        self.operational_combinations = [('AT Engaged', 'AP Engaged', 'Airborne',)]
        self.can_operate_kwargs = {'ac_family': A('Family', value='B737 NG')}

    def test_derive(self):
        ap_engaged = M('AP Engaged', array=np.ma.array([1]*40), values_mapping={0: '-', 1: 'Engaged'})
        at_engaged = M('AT Engaged', array=np.ma.array([1]*5+ [0]*30 + [1]*5), values_mapping={0: '-', 1: 'Engaged'})
        airs = buildsection('Airborne', 1, 39)

        node = self.node_class()
        node.derive(at_engaged, ap_engaged, airs)

        name = 'AT Disengaged AP Engaged Duration'
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=5, value=30),
        ])
        self.assertEqual(node, expected)


class TestNumberOfAPChannelsEngagedAtTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = NumberOfAPChannelsEngagedAtTouchdown
        self.operational_combinations = [('Touchdown', 'AP Engaged', 'AP Channels Engaged'), ('Touchdown', 'AP Engaged'), ('Touchdown', 'AP Channels Engaged')]
        self.tdwn = KTI('Touchdown', items=[KeyTimeInstance(3, 'Touchdown')])

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)
        for combination in self.operational_combinations:
            self.assertTrue(combination in opts)

    def test_derive_ap_engaged(self):
        ap_engaged = M('AP Engaged', np.ma.array([1,1,1,1,1,1]), values_mapping={0: '-', 1: 'Engaged'})
        node = self.node_class()
        node.derive(self.tdwn, ap_engaged, None)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 1)

    def test_derive_channel_count(self):
        mapping = {0: '-',
                   1: 'Single',
                   2: 'Dual',
                   3: 'Triple',}
        ap_ch_count = M('AP Channels Engaged', np.ma.array([3,3,3,3,3]), values_mapping=mapping)
        node = self.node_class()
        node.derive(self.tdwn, None, ap_ch_count)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 3)


##############################################################################

class TestThrustReversersDeployedDuration(unittest.TestCase):
    def test_can_operate(self):
        ops = ThrustReversersDeployedDuration.get_operational_combinations()
        self.assertEqual(ops, [('Thrust Reversers', 'Landing')])

    def test_derive(self):
        rev = M(array=np.ma.zeros(30), values_mapping={
            0: 'Stowed', 1: 'In Transit', 2: 'Deployed',}, frequency=2)
        ldg = S(frequency=2)
        ldg.create_section(slice(5, 25))
        # no deployment
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 0)

        # deployed for a while
        rev.array[6:13] = 'Deployed'
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(dur[0].index, 5.5)
        self.assertEqual(dur[0].value, 3.5)

        # deployed the whole time
        rev.array[:] = 'Deployed'
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(len(dur), 1)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 10)


class TestThrustRatingCLB1Duration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustRatingCLB1Duration
        self.operational_combinations = [('Thrust Rating Mode', 'Airborne')]

    def test_derive(self):
        array = np.ma.zeros(10)
        thrust = M(array=array, values_mapping={0: '-', 1: 'CLB1'})
        airs = buildsection('Airborne', 3, 9)
        node = self.node_class()
        node.derive(thrust, airs)
        self.assertEqual(len(node), 0)

        array[2:5] = 1
        array[7:8] = 1
        node = self.node_class()
        node.derive(thrust, airs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(node[1].index, 7)
        self.assertEqual(node[1].value, 1)


class TestThrustReversersDeployedDuringFlightDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = ThrustReversersDeployedDuringFlightDuration

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        self.assertEqual(ops, [('Thrust Reversers', 'Airborne')])

    def test_derive(self):
        rev = M(array=np.ma.zeros(30), values_mapping={
            0: 'Stowed', 1: 'In Transit', 2: 'Deployed',}, frequency=2)
        airs = S(frequency=2)
        airs.create_section(slice(5, 25))
        # no deployment
        dur = self.node_class()
        dur.derive(rev, airs)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 0)

        # deployed for a while
        rev.array[6:13] = 'Deployed'
        dur = self.node_class()
        dur.derive(rev, airs)
        self.assertEqual(dur[0].index, 5.5)
        self.assertEqual(dur[0].value, 3.5)

        # deployed the whole time
        rev.array[:] = 'Deployed'
        dur = self.node_class()
        dur.derive(rev, airs)
        self.assertEqual(len(dur), 1)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 10)


class TestThrustReversersCancelToEngStopDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustReversersCancelToEngStopDuration
        self.operational_combinations = [('Thrust Reversers', 'Eng Start', 'Eng Stop')]

    def test_derive(self):
        thrust_reversers = load(os.path.join(
            test_data_path,
            'ThrustReversersCancelToEngStopDuration_ThrustReversers_1.nod'))
        eng_start = KTI('Eng Start', items=[
            KeyTimeInstance(10, 'Eng Start'),
        ])
        eng_stop = load(os.path.join(
            test_data_path,
            'ThrustReversersCancelToEngStopDuration_EngStop_1.nod'))
        node = ThrustReversersCancelToEngStopDuration()
        node.derive(thrust_reversers, eng_start, eng_stop)
        self.assertEqual(node, [KeyPointValue(2920.60546875, 267.10546875, 'Thrust Reversers Cancel To Eng Stop Duration')])


class TestTouchdownToThrustReversersDeployedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TouchdownToThrustReversersDeployedDuration
        self.operational_combinations = [('Thrust Reversers', 'Landing', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownToSpoilersDeployedDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = TouchdownToSpoilersDeployedDuration
        self.values_mapping = {0: 'Stowed', 1: 'Armed/Cmd Dn', 2: 'Deployed/Cmd Up'}

    def test_one_landing_one_spdbrake_deployement(self):
        speedbrake_array =  np.ma.array([1] * 10 + [2] * 3 + [0] * 2)
        speedbrake = M('Speedbrake Selected', values_mapping=self.values_mapping,
                       array=speedbrake_array)
        lands = buildsection('Landing', 0, 15)
        tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(5, 'Touchdown'),
        ])
        node = self.node_class()
        node.derive(speedbrake, lands, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 4.5)
        self.assertEqual(node[0].index, 9.5)

    def test_one_landing_two_spdbrake_deployements(self):
        speedbrake_array =  np.ma.array([1] * 7 + [2] * 3 + [1] * 2 + [2] * 3)
        speedbrake = M('Speedbrake Selected', values_mapping=self.values_mapping,
                       array=speedbrake_array)
        lands = buildsection('Landing', 0, 15)
        tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(5, 'Touchdown'),
        ])
        node = self.node_class()
        node.derive(speedbrake, lands, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1.5)
        self.assertEqual(node[0].index, 6.5)

    def test_two_landing_phases(self):
        speedbrake_array =  np.ma.array(([1] * 10 + [2] * 3 + [0] * 2) * 2)
        speedbrake = M('Speedbrake Selected', values_mapping=self.values_mapping,
                       array=speedbrake_array)
        lands = buildsections('Landing', [0, 10], [15, 30])
        tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(5, 'Touchdown'),
            KeyTimeInstance(20, 'Touchdown'),
        ])
        node = self.node_class()
        node.derive(speedbrake, lands, tdwns)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 4.5)
        self.assertEqual(node[0].index, 9.5)
        self.assertEqual(node[1].value, 4.5)
        self.assertEqual(node[1].index, 24.5)

    def test_no_speedbrake_deployement(self):
        speedbrake_array =  np.ma.array([1] * 15)
        speedbrake = M('Speedbrake Selected', values_mapping=self.values_mapping,
                       array=speedbrake_array)
        lands = buildsection('Landing', 0, 15)
        tdwns = KTI('Touchdown', items=[
            KeyTimeInstance(5, 'Touchdown'),
        ])
        node = self.node_class()
        node.derive(speedbrake, lands, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 0.0)
        self.assertEqual(node[0].index, 0)


class TestSpoilersDeployedDurationDuringLanding(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpoilersDeployedDurationDuringLanding
        self.values_mapping = {0: 'Stowed', 1: 'Armed/Cmd Dn', 2: 'Deployed/Cmd Up'}
        self.speedbrake_array =  np.ma.array(([0] * 4 + [1] * 2 + [2] * 4) * 3)
        self.speedbrake = M('Speedbrake Selected', values_mapping=self.values_mapping,
                            array=self.speedbrake_array)
        self.landing = S('Landing')
        self.landing.create_section(slice(5, 28, None),
                                    begin=5.265625, end=27.265625)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertIn('Speedbrake Selected', opts[0])
        self.assertIn('Landing', opts[0])

    def test_derive(self):
        node = self.node_class()
        node.derive(self.speedbrake, self.landing)
        self.assertEqual(node, [KeyPointValue(27.265625, 9.265625,
                                              'Spoilers Deployed Duration During Landing')])

    def test_derive_multiple_landings(self):
        node = self.node_class()
        self.landing.create_section(slice(32, 40, None),
                                    begin=32.2505, end=40.331)
        self.speedbrake_array = np.concatenate((self.speedbrake_array, self.speedbrake_array))
        self.speedbrake.array = self.speedbrake_array
        node.derive(self.speedbrake, self.landing)
        self.assertEqual(node[0], KeyPointValue(27.265625, 9.265625,
                                                'Spoilers Deployed Duration During Landing'))
        self.assertEqual(node[1], KeyPointValue(40.331, 4.0,
                                                'Spoilers Deployed Duration During Landing'))

    def test_derive_landing_without_spoilers_deployed(self):
        node = self.node_class()
        self.speedbrake_array = np.ma.array(([0] * 4 + [1] * 2 + [0] * 4) * 3)
        self.speedbrake.array = self.speedbrake_array
        node.derive(self.speedbrake, self.landing)
        self.assertEqual(node, [KeyPointValue(27.265625, 0,
                                              'Spoilers Deployed Duration During Landing')])


##############################################################################
# TOGA Usage


class TestTOGASelectedDuringFlightDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TOGASelectedDuringFlightDuration
        self.operational_combinations = [('Takeoff And Go Around', 'Go Around And Climbout', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTOGASelectedDuringGoAroundDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TOGASelectedDuringGoAroundDuration
        self.operational_combinations = [('Takeoff And Go Around', 'Go Around And Climbout')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestLiftoffToClimbPitchDuration(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Landing Gear


class TestMainGearOnGroundToNoseGearOnGroundDuration(unittest.TestCase,
                                                     NodeTest):

    def test_derive(self):
        self.node_class = MainGearOnGroundToNoseGearOnGroundDuration
        self.operational_combinations = [('Brake Pressure', 'Takeoff Roll',)]
        self.function = max_value

        gog_array = np.ma.concatenate((np.zeros(20), np.ones(15)))
        gog = M(
            name='Gear On Ground',
            array=gog_array,
            values_mapping={0: 'Air', 1: 'Ground'},
        )
        gogn_array = np.ma.concatenate((np.zeros(25), np.ones(10)))
        gogn = M(
            name='Gear (N) On Ground',
            array=gogn_array,
            values_mapping={0: 'Air', 1: 'Ground'},
        )
        landing = buildsection('Landing', 10, 30)
        node = self.node_class()
        node.derive(gog, gogn, landing)
        self.assertEqual(node, [
            KeyPointValue(
                19.5, 5.0,
                'Main Gear On Ground To Nose Gear On Ground Duration'),
        ])


##################################
# Braking


class TestBrakeTempDuringTaxiInMax(unittest.TestCase,
                                   CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = BrakeTempDuringTaxiInMax
        self.operational_combinations = [('Brake (*) Temp Max', 'Taxi In',)]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestBrakeTempBeforeTakeoffMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = BrakeTempBeforeTakeoffMax
        self.operational_combinations = [('Brake (*) Temp Max', 'Taxi Out',
                                          'Liftoff')]

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(280, 316, 1), np.ma.arange(317, 260, -2)))

        brake_temp = P('Brake (*) Temp Max', array)
        liftoff = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='liftoff', index=30),
        ])
        takiout = buildsection('Taxi Out', 2, 35)

        node = self.node_class()
        node.derive(brake_temp, takiout, liftoff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 29)
        self.assertEqual(node[0].value, 309)


class TestBrakeTempAfterTouchdownDelta(unittest.TestCase):
    def setUp(self):
        self.node_class = BrakeTempAfterTouchdownDelta
        self.operational_combinations = [('Brake (*) Temp Avg', 'Touchdown',)]

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         self.operational_combinations)

    def test_derive_basic(self):
        array = np.ma.concatenate((np.arange(200, 130, -1), np.arange(130, 300, 10), np.arange(300, 280, -1)))

        brake_temp = P('Brake (*) Temp Avg', array)
        touchdown = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=70),
        ])

        node = self.node_class()
        node.derive(brake_temp, touchdown)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 87)
        self.assertEqual(node[0].value, 170)

    def test_basic(self):
        # DJ wrote this test, having implemented the same algorithm
        # independently. Transferred here to show great minds think alike.
        temp=P('Brake (*) Temp Avg', np.ma.array(data=[3,3,3,4,5,6,7,8,9,8,7]))
        touchdown = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=2),])
        dt = BrakeTempAfterTouchdownDelta()
        dt.get_derived((temp, touchdown))
        self.assertEqual(dt[0].value, 6)
        self.assertEqual(dt[0].index, 8)

class TestBrakePressureInTakeoffRollMax(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = BrakePressureInTakeoffRollMax
        self.operational_combinations = [('Brake Pressure',
                                          'Takeoff Roll Or Rejected Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDelayedBrakingAfterTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DelayedBrakingAfterTouchdown
        self.operational_combinations = [('Landing', 'Groundspeed', 'Touchdown')]

    def test_derive_basic(self):
        array = np.ma.arange(10, 0, -0.05) ** 3 / 10
        groundspeed = P('Groundspeed', np.ma.concatenate((array, array)))
        landing = buildsections('Landing', (10, 100), (210, 400))
        touchdown = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=15),
            KeyTimeInstance(name='Touchdown', index=210),
        ])

        node = self.node_class()
        node.derive(landing, groundspeed, touchdown)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=84.7, value=61.6),
            KeyPointValue(name=name, index=272.8, value=55.1),
        ])

        self.assertEqual(node.name, expected.name)
        self.assertEqual(len(node), len(expected))
        for a, b in zip(node, expected):
            self.assertAlmostEqual(a.index, b.index, delta=0.1)
            self.assertAlmostEqual(a.value, b.value, delta=0.1)


class TestDecelerationAfterTouchdownAvg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DecelerationAfterTouchdownAvg
        self.operational_combinations = [('Landing', 'Groundspeed', 'Touchdown')]

    def test_derive_basic(self):
        array = np.ma.arange(10, 0, -0.05) ** 3 / 10 + 50
        groundspeed = P('Groundspeed', np.ma.concatenate((array, array)))
        landing = buildsections('Landing', (0, 100), (210, 400))
        touchdown = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=15),
            KeyTimeInstance(name='Touchdown', index=241),
        ])

        node = self.node_class()
        node.derive(landing, groundspeed, touchdown)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=84.7, value=50.0 / 61.6),
            KeyPointValue(name=name, index=307.2, value=30.25 / 54.84),
        ])

        self.assertEqual(node.name, expected.name)
        self.assertEqual(len(node), len(expected))
        for a, b in zip(node, expected):
            self.assertAlmostEqual(a.index, b.index, delta=0.1)
            self.assertAlmostEqual(a.value, b.value, delta=0.1)

    def test_too_low_td_speed(self):
        array = np.ma.arange(80.0, 10.0, -0.5)
        groundspeed = P('Groundspeed', array)
        landing = buildsection('Landing', 0, 100)
        touchdown = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=20),
        ])

        node = self.node_class()
        node.derive(landing, groundspeed, touchdown)

        self.assertEqual(len(node), 0)


class TestAutobrakeRejectedTakeoffNotSetDuringTakeoff(unittest.TestCase,
                                                      CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Autobrake Selected RTO'
        self.phase_name = 'Takeoff Roll Or Rejected Takeoff'
        self.node_class = AutobrakeRejectedTakeoffNotSetDuringTakeoff
        self.values_mapping = {0: '-', 1: 'Selected'}

        self.values_array = np.ma.concatenate((np.zeros(5), np.ones(4), np.zeros(3)))
        self.expected = [KeyPointValue(
            index=2, value=3.0,
            name='Autobrake Rejected Takeoff Not Set During Takeoff')]

        self.basic_setup()


    def test_with_masked_values(self):
        node = AutobrakeRejectedTakeoffNotSetDuringTakeoff()

        # Masked values are considered "correct" in given circumstances, in
        # this case we assume them to be "Selected"
        autobrake = M('Autobrake Selected RTO',
                      np.ma.array(
                          [0] * 5 + [1] * 4 + [0] * 3,
                          mask=[False] * 3 + [True] * 2 + [False] * 7),
                      values_mapping = {0: '-', 1: 'Selected'})
        phase = S('Takeoff Roll Or Rejected Takeoff')
        phase.create_section(slice(2, 7))
        node.derive(autobrake, phase)
        self.assertEqual(node, [KeyPointValue(
            index=2, value=1.0,
            name='Autobrake Rejected Takeoff Not Set During Takeoff')])



##############################################################################
# Alpha Floor


class TestAlphaFloorDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AlphaFloorDuration
        self.operational_combinations = [
            ('Alpha Floor', 'Airborne'),
            ('FMA AT Information', 'Airborne'),
            ('Alpha Floor', 'FMA AT Information', 'Airborne'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented.')

    def test_derive_basic(self):
        array = np.ma.concatenate((np.ones(1), np.zeros(2), np.ones(3), np.zeros(14)))
        mapping = {0: '-', 1: 'Engaged'}
        alpha_floor = M('Alpha Floor', array=array, values_mapping=mapping)

        array = np.ma.repeat([0, 0, 1, 2, 3, 4, 3, 2, 1, 0], 2)
        mapping = {0: '-', 1: '-', 2: '-', 3: 'Alpha Floor', 4: '-' }
        autothrottle_info = M('FMA AT Information', array=array, values_mapping=mapping)

        airs = buildsection('Airborne', 2, 18)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=3, value=3),
            KeyPointValue(name=name, index=8, value=2),
            KeyPointValue(name=name, index=12, value=2),
        ])

        node = self.node_class()
        node.derive(alpha_floor, autothrottle_info, airs)
        self.assertEqual(node, expected)


##############################################################################
# Altitude


########################################
# Altitude: General


class TestAltitudeMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AltitudeMax
        self.operational_combinations = [('Altitude STD Smoothed', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeDuringGoAroundMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AltitudeDuringGoAroundMin
        self.operational_combinations = [('Altitude AAL', 'Go Around And Climbout')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeOvershootAtSuspectedLevelBust(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeOvershootAtSuspectedLevelBust
        self.operational_combinations = [('Altitude STD Smoothed', 'Altitude AAL')]

    def test_derive_too_slow(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array(1.0 + np.sin(np.arange(0, 12.6, 0.1))) * 1000,
            frequency=0.02,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.get_derived([alt_std, alt_std])
        self.assertEqual(node, [])

    def test_derive_straight_up_and_down(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.concatenate((np.arange(0, 10000, 50), np.arange(10000, 0, -50))),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std, alt_std)
        self.assertEqual(node, [])

    def test_derive_up_and_down_with_overshoot(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.concatenate((np.arange(0, 9800, 50), np.arange(9800, 9000, -50), np.ones(200) * 9000, np.arange(9000, 0, -50))),
            frequency=0.25,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.get_derived([alt_std, alt_std])
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 196)
        self.assertEqual(node[0].value, 800)

    def test_derive_up_and_down_with_undershoot(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.concatenate((
                np.arange(0, 9800, 50),
                np.ones(200) * 9800,
                np.arange(9800, 9000, -50),
                np.arange(9000, 20000, 50),
                np.arange(20000, 0, -50))),
            frequency=0.25,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.get_derived([alt_std, alt_std])
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 412)
        self.assertEqual(node[0].value, -800)

    def test_derive_with_real_go_around_data_ignores_undershoot(self):
        '''
        Undershoots under 3000 ft are excluded due to different Go Around behaviour.
        '''
        alt_std = load(os.path.join(test_data_path,
                                    'alt_std_smoothed_go_around.nod'))
        bust = AltitudeOvershootAtSuspectedLevelBust()
        bust.derive(alt_std, alt_std)
        self.assertEqual(len(bust), 0)

    def test_derive_real_data_overshoot(self):
        '''
        Undershoots under 3000 ft are excluded due to different Go Around behaviour.
        '''
        alt_std = load(os.path.join(test_data_path,
                                    'alt_overshoot_alt_std.nod'))
        bust = AltitudeOvershootAtSuspectedLevelBust()
        bust.derive(alt_std, alt_std)
        self.assertEqual(len(bust), 1)
        self.assertAlmostEqual(bust[0].index, 3713, places=0)
        self.assertAlmostEqual(bust[0].value, 543, places=0)

    def test_derive_no_duplicates_1(self):
        node = AltitudeOvershootAtSuspectedLevelBust()
        alt_std = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_std_01.nod'))
        alt_aal = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_aal_01.nod'))
        node.derive(alt_std, alt_aal)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 781, places=0)
        self.assertAlmostEqual(node[0].value, 533, places=0)

    def test_derive_no_duplicates_2(self):
        node = AltitudeOvershootAtSuspectedLevelBust()
        alt_std = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_std_02.nod'))
        alt_aal = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_aal_02.nod'))
        node.derive(alt_std, alt_aal)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 1192, places=0)
        self.assertAlmostEqual(node[0].value, 330, places=0)

    def test_derive_overshoot_during_descent(self):
        node = AltitudeOvershootAtSuspectedLevelBust()
        alt_std = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_std_03.nod'))
        alt_aal = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_aal_03.nod'))
        node.derive(alt_std, alt_aal)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 2478, places=0)
        self.assertAlmostEqual(node[0].value, -418, places=0)

    def test_derive_level_flights(self):
        node = AltitudeOvershootAtSuspectedLevelBust()
        alt_std = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_std_04.nod'))
        alt_aal = load(os.path.join(test_data_path, 'AltitudeOvershootAtSuspectedLevelBust_alt_aal_04.nod'))
        node.derive(alt_std, alt_aal)
        self.assertEqual(len(node), 0)

    def test_derive_over_900ft(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.concatenate((
                np.arange(0, 10000, 50),
                np.arange(10000, 9000, -50),
                np.ones(200) * 9000,
                np.arange(9000, 0, -50))),
            frequency=0.25,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.get_derived([alt_std, alt_std])
        self.assertEqual(len(node), 0)


class TestAltitudeSTDMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeSTDMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Altitude STD',)])

    def test_derive(self):
        alt_std = P('Altitude STD', np.ma.arange(0, 11))
        node = self.node_class()
        node.derive(alt_std)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)


class TestAltitudeInCruiseAverage(unittest.TestCase):
    def test_can_operate(self):
        combinations = AltitudeInCruiseAverage.get_operational_combinations()
        self.assertTrue(('Altitude STD','Cruise') in combinations)

    def test_basic(self):
        alt = P('Altitude STD', array=np.ma.concatenate((np.ones(50) * 25000, np.ones(50) * 35000)))
        cruises = buildsections('Cruise', [5, 35], [65,95])
        spd = AltitudeInCruiseAverage()
        spd.get_derived((alt, cruises))
        self.assertAlmostEqual(spd[0].value, 30000)

####class TestAltitudeAtCabinPressureLowWarningDuration(unittest.TestCase,
####                                                    CreateKPVsWhereTest):
####    def setUp(self):
####        self.param_name = 'Cabin Altitude'
####        self.phase_name = 'Airborne'
####        self.node_class = AltitudeAtCabinPressureLowWarningDuration
####        self.values_mapping = {0: '-', 1: 'Warning'}
####
####        self.basic_setup()


########################################
# Altitude: Flap


class TestAltitudeWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeWithFlapMax
        self.operational_combinations = [
            ('Flap Lever', 'Altitude STD Smoothed', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Altitude STD Smoothed', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Altitude STD Smoothed', 'Airborne'),
        ]

    def test_derive(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.arange(0, 3000, 100),
        )
        airborne = buildsection('Airborne', 2, 28)
        name = self.node_class.get_name()

        array = np.ma.repeat((0, 1, 5, 15, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(alt_std, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=28, value=2800, name=name),
        ]))

        array = np.ma.repeat((1, 2, 5, 15, 25, 30), 5)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

        node = self.node_class()
        node.derive(alt_std, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=28, value=2800, name=name),
        ]))


class TestAltitudeAtFlapExtension(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFlapExtension
        self.operational_combinations = [('Flap Extension While Airborne', 'Altitude AAL')]
        self.alt_aal = P('Altitude AAL', np.ma.concatenate((np.ones(15) * 1234.0, np.ones(15) * 2345.0)))

    def test_derive_multiple_ktis(self):
        flap_exts = KTI(name='Flap Extension While Airborne', items=[
            KeyTimeInstance(10, 'Flap Extension While Airborne'),
            KeyTimeInstance(20, 'Flap Extension While Airborne'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_exts)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=1234, name=name),
            KeyPointValue(index=20, value=2345, name=name),
        ]))

    def test_derive_no_ktis(self):
        flap_exts = KTI(name='Flap Extension While Airborne', items=[])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_exts)
        self.assertEqual(node, KPV(name=name, items=[]))


class TestAltitudeAtFirstFlapExtensionAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapExtensionAfterLiftoff
        self.operational_combinations = [('Altitude At Flap Extension',)]

    def test_derive_basic(self):
        flap_exts = KPV(name='Altitude At Flap Extension', items=[
            KeyPointValue(index=7, value=21, name='Altitude At Flap Extension'),
            KeyPointValue(index=14, value=43, name='Altitude At Flap Extension'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap_exts)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7, value=21, name=name),
        ]))


class TestAltitudeAtFirstFlapChangeAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapChangeAfterLiftoff
        self.operational_combinations = [
            ('Flap Lever', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
        ]

    def test_derive(self):
        array = np.ma.array((0, 5, 5, 5, 5, 0, 0, 0, 0, 15, 30, 30, 30, 30, 15, 0))
        mapping = {0: '0', 5: '5', 15: '15', 30: '30'}
        flap_synth = M('Flap Lever (Synthetic)', array, values_mapping=mapping)
        flap_takeoff = KPV('Flap At Liftoff', items=[
            KeyPointValue(name='Flap At Liftoff', index=2, value=5.0),
        ])
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array, alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airborne = buildsection('Airborne', 2, 14)

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, None, flap_synth, flap_takeoff, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=4.5, value=150, name=name),
        ]))

    def test_derive_no_flap_takeoff(self):
        array = np.ma.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 30, 30, 30, 30, 15, 0))
        mapping = {0: '0', 5: '5', 15: '15', 30: '30'}
        flap_synth = M('Flap Lever (Synthetic)', array, values_mapping=mapping)
        flap_takeoff = KPV('Flap At Liftoff', items=[
            KeyPointValue(name='Flap At Liftoff', index=2, value=0.0),
        ])
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array,alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airborne = buildsection('Airborne', 2, 14)

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, None, flap_synth, flap_takeoff, airborne)
        self.assertEqual(node, KPV(name=name, items=[]))


class TestAltitudeAtFlapExtensionWithGearDownSelected(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFlapExtensionWithGearDownSelected
        self.operational_combinations = [
            ('Flap Lever', 'Altitude AAL', 'Gear Down Selected', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Altitude AAL', 'Gear Down Selected', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Altitude AAL', 'Gear Down Selected', 'Airborne'),
        ]

    def test_derive(self):
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 22: '22', 30: '30'}
        array = np.ma.array((0, 5, 5, 0, 0, 0, 1, 1, 10, 22, 22, 22, 30, 30, 15, 0.0))
        flap_synth = M('Flap Lever (Synthetic)', array, values_mapping=mapping)

        array = np.ma.array((0, 0, 0, 50, 100, 200, 300, 400))
        alt_aal = P('Altitude AAL', np.ma.concatenate((array, array[::-1])))

        gear = M('Gear Down Selected', np.ma.concatenate((np.zeros(7), np.ones(8))),
                 values_mapping={0:'Up', 1:'Down'})
        airborne = buildsection('Airborne', 2, 14)

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, None, flap_synth, gear, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8, value=400, name='Altitude At Flap 10 Extension With Gear Down Selected'),
            KeyPointValue(index=9, value=300, name='Altitude At Flap 22 Extension With Gear Down Selected'),
            KeyPointValue(index=12, value=50, name='Altitude At Flap 30 Extension With Gear Down Selected'),
        ]))


class TestAirspeedAtFlapExtensionWithGearDownSelected(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAtFlapExtensionWithGearDownSelected
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed', 'Gear Down Selected', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Airspeed', 'Gear Down Selected', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airspeed', 'Gear Down Selected', 'Airborne'),
        ]

    def test_derive(self):
        array = np.ma.array((0, 5, 5, 0, 0, 0, 1, 1, 10, 22, 22, 22, 30, 30, 15, 0.0))
        mapping = {0: '0', 1: '1', 5: '5', 10: '10', 15: '15', 22: '22', 30: '30'}
        flap_synth = M('Flap Lever (Synthetic)', array, values_mapping=mapping)

        array = np.ma.array((0, 0, 0, 50, 100, 200, 250, 280))
        air_spd = P('Airspeed', np.ma.concatenate((array, array[::-1])))

        gear = M('Gear Down Selected', np.ma.concatenate((np.zeros(7), np.ones(8))),
                 values_mapping={0:'Up', 1:'Down'})
        airborne = buildsection('Airborne', 2, 14)

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(air_spd, None, flap_synth, gear, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8, value=280, name='Airspeed At Flap 10 Extension With Gear Down Selected'),
            KeyPointValue(index=9, value=250, name='Airspeed At Flap 22 Extension With Gear Down Selected'),
            KeyPointValue(index=12, value=50, name='Airspeed At Flap 30 Extension With Gear Down Selected'),
        ]))


class TestAltitudeAALCleanConfigurationMin(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeAALCleanConfigurationMin
        self.operational_combinations = [
            ('Altitude AAL', 'Flap', 'Gear Retracted'),
        ]

    def test_derive(self):
        array = np.ma.concatenate((np.ones(8) * 15, np.zeros(9), np.ones(8) * 15)).astype(np.int32)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M('Flap', array, values_mapping=mapping)

        array = np.ma.concatenate((
            np.arange(0, 1000, 100),
            np.ones(5) * 1000,
            np.arange(1000, 0, -100),
        ))
        alt_rad = P('Altitude AAL', array=array)

        gear_retr = S(items=[Section('Gear Retracted', slice(5, 10), 5, 10)])

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_rad, flap, gear_retr)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8.0, value=800.0, name=name),
        ]))


class TestAltitudeAtLastFlapChangeBeforeTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtLastFlapChangeBeforeTouchdown
        self.operational_combinations = [
            ('Flap Lever', 'Altitude AAL', 'Touchdown'),
            ('Flap Lever (Synthetic)', 'Altitude AAL', 'Touchdown'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Altitude AAL', 'Touchdown'),
        ]

    def test_derive(self):
        array = np.ma.concatenate((np.ones(8) * 10, np.ones(7) * 15))
        mapping = {0: '0', 10: '10', 15: '15'}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.concatenate((np.arange(1000, 0, -100), np.zeros(5)))
        alt_aal = P(name='Altitude AAL', array=array)
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10)])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, flap_lever, None, touchdowns, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8.0, value=200.0, name=name),
        ]))

    def test_late_retraction(self):
        array = np.ma.concatenate((np.ones(8) * 15, np.ones(7) * 10))
        mapping = {0: '0', 10: '10', 15: '15'}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.concatenate((np.arange(1000, 0, -100), np.zeros(5)))
        alt_aal = P(name='Altitude AAL', array=array)
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10)])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, flap_lever, None, touchdowns, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8.0, value=200.0, name=name),
        ]))

    def test_derive_auto_retract_after_touchdown(self):
        array = np.ma.concatenate((np.ones(6) * 10, np.ones(3) * 15, np.ones(6) * 10))
        mapping = {0: '0', 10: '10', 15: '15'}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.concatenate((np.arange(1000, 0, -100), np.zeros(5)))
        alt_aal = P(name='Altitude AAL', array=array)
        far = M('Flap Automatic Retraction',
                array=np.ma.array([0]*10+[1]*5),
                values_mapping={0:'-', 1:'Retract'})
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10)])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, flap_lever, None, touchdowns, far)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=6.0, value=400.0, name=name),
        ]))

    def test_derive_auto_retract_before_touchdown(self):
        array = np.ma.concatenate((np.ones(6) * 10, np.ones(3) * 15, np.ones(6) * 10))
        mapping = {0: '0', 10: '10', 15: '15'}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.concatenate((np.ma.arange(1000, 0, -100), np.zeros(5)))
        alt_aal = P(name='Altitude AAL', array=array)
        far = M('Flap Automatic Retraction',
                array=np.ma.concatenate((np.zeros(9), np.ones(6))),
                values_mapping={0:'-', 1:'Retract'})
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10)])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, flap_lever, None, touchdowns, far)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=6.0, value=400.0, name=name),
        ]))


class TestAltitudeAtLastFlapSelectionBeforeTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AltitudeAtLastFlapSelectionBeforeTouchdown
        self.alt_aal=P('Altitude AAL', np.ma.array(np.linspace(1000, 0, 30)))
        self.tdwns=KTI('Touchdown', items=[KeyTimeInstance(name='Touchdown',
                                                           index=27),])
        # Flap Lever - expecting 3 KPVs from this parameter, 15, 30 and 35.
        flap_array = np.ma.array([
            0, 5, 15, 15, 15, 0, 0, 0, 0, 0,
            0, 0,  0,  0,  0, 0, 0, 0, 0, 0,
            0, 15, 15, 15, 30, 35, 35, 15, 15, 15
        ])
        flap_mapping = {int(f): str(f) for f in np.ma.unique(flap_array)}
        self.flap = M('Flap Lever', flap_array, values_mapping=flap_mapping)

        # Synth Lever - expecting 2 KPVs from this parameter, 15 and 35.
        synth_array = np.ma.array([
            0, 5, 15, 15, 15, 0, 0, 0, 0, 0,
            0, 0,  0,  0,  0, 0, 0, 0, 0, 0,
            0, 15, 15, 15, 35, 35, 35, 30, 30, 30
        ])
        synth_mapping = {int(f): 'Lever %s' % i for i, f in enumerate(
            np.ma.unique(synth_array))}
        self.synth = M('Flap Lever (Synthetic)', synth_array,
                       values_mapping=synth_mapping)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)
        for opt in opts:
            self.assertIn('Altitude AAL', opt)
            self.assertIn('Touchdown', opt)
            self.assertTrue(any_of(['Flap Lever', 'Flap Lever (Synthetic)'],
                                   opt))

    def test_derive(self):
        node = self.node_class()
        node.derive(self.alt_aal, self.flap, self.synth, self.tdwns)

        self.assertEqual(len(node), 3)
        self.assertAlmostEqual(node[0].value, 275.86, places=2)
        self.assertEqual(node[0].index, 21)
        self.assertEqual(node[0].name,
                         'Altitude At Last Flap 15 Selection Before Touchdown')
        self.assertAlmostEqual(node[1].value, 172.41, places=2)
        self.assertEqual(node[1].index, 24)
        self.assertEqual(node[1].name,
                         'Altitude At Last Flap 30 Selection Before Touchdown')
        self.assertAlmostEqual(node[2].value, 137.93, places=2)
        self.assertEqual(node[2].index, 25)
        self.assertEqual(node[2].name,
                         'Altitude At Last Flap 35 Selection Before Touchdown')

    def test_derive_synth_only(self):
        '''
        Shouldn't a KPV for 30 deg as flap lever goes down from 35 to 30 deg
        '''
        node = self.node_class()
        node.derive(self.alt_aal, None, self.synth, self.tdwns)

        self.assertEqual(len(node), 2)
        self.assertAlmostEqual(node[0].value, 275.86, places=2)
        self.assertEqual(node[0].index, 21)
        self.assertEqual(node[0].name,
                         'Altitude At Last Flap 15 Selection Before Touchdown')
        self.assertAlmostEqual(node[1].value, 172.41, places=2)
        self.assertEqual(node[1].index, 24)
        self.assertEqual(node[1].name,
                         'Altitude At Last Flap 35 Selection Before Touchdown')


class TestAltitudeAtFirstFlapRetractionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapRetractionDuringGoAround
        self.operational_combinations = [
            ('Altitude AAL', 'Flap Retraction During Go Around', 'Go Around And Climbout'),
        ]
        self.alt_aal = P(
            name='Altitude AAL',
            array=np.ma.concatenate([
                np.zeros(10),
                np.arange(40) * 1000,
                np.ones(10) * 40000,
                np.arange(40, 0, -1) * 1000,
                np.arange(1, 3) * 1000,
                np.ones(10) * 3000,
                np.arange(3, -1, -1) * 1000,
                np.zeros(10),
                ]),
        )
        self.go_arounds = buildsection('Go Around And Climbout', 97, 112)

    def test_derive_multiple_ktis(self):
        flap_rets = KTI(name='Flap Retraction During Go Around', items=[
            KeyTimeInstance(index=100, name='Flap Retraction During Go Around'),
            KeyTimeInstance(index=104, name='Flap Retraction During Go Around'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets, self.go_arounds)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=100, value=0, name=name),
        ]))

    def test_derive_no_ktis(self):
        flap_rets = KTI(name='Flap Retraction During Go Around', items=[])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets, self.go_arounds)
        self.assertEqual(node, KPV(name=name, items=[]))


class TestAltitudeAtFirstFlapRetraction(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapRetraction
        self.operational_combinations = [('Altitude AAL', 'Flap Retraction While Airborne')]
        self.alt_aal = P(
            name='Altitude AAL',
            array=np.ma.concatenate([
                np.zeros(10),
                np.arange(40) * 1000,
                ]),
        )

    def test_derive_basic(self):
        flap_rets = KTI(name='Flap Retraction While Airborne', items=[
            KeyTimeInstance(index=30, name='Flap Retraction While Airborne'),
            KeyTimeInstance(index=40, name='Flap Retraction While Airborne'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=30, value=20000, name=name),
        ]))

    def test_derive_no_ktis(self):
        flap_rets = KTI(name='Flap Retraction While Airborne', items=[])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, KPV(name=name, items=[]))


class TestAltitudeAtLastFlapRetraction(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtLastFlapRetraction
        self.operational_combinations = [('Altitude AAL', 'Flap Retraction While Airborne')]
        self.alt_aal = P(
            name='Altitude AAL',
            array=np.ma.concatenate([
                np.zeros(10),
                np.arange(40) * 1000,
                ]),
        )

    def test_derive_basic(self):
        flap_rets = KTI(name='Flap Retraction While Airborne', items=[
            KeyTimeInstance(index=30, name='Flap Retraction While Airborne'),
            KeyTimeInstance(index=40, name='Flap Retraction While Airborne'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=40, value=30000, name=name),
        ]))

    def test_derive_no_ktis(self):
        flap_rets = KTI(name='Flap Retraction While Airborne', items=[])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, KPV(name=name, items=[]))


class TestAltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft
        self.operational_combinations = [('Altitude AAL',
                                          'Climb Thrust Derate Deselected',
                                          'Climbing')]

    def test_derive_basic(self):
        alt_aal_array = np.ma.concatenate(
            [np.arange(0, 40000, 4000), np.ones(10) * 40000,
             np.arange(40000, 0, -4000)])
        alt_aal = P('Altitude AAL', array=alt_aal_array)
        climb_thrust_derate = KTI('Climb Thrust Derate Deselected', items=[
            KeyTimeInstance(5, 'Climb Thrust Derate Deselected'),
            KeyTimeInstance(12, 'Climb Thrust Derate Deselected'),
            KeyTimeInstance(17, 'Climb Thrust Derate Deselected')])
        climbs = buildsection('Climbing', 0, 14)
        node = self.node_class()
        node.derive(alt_aal, climb_thrust_derate, climbs)
        self.assertEqual(node, [
            KeyPointValue(5, 20000.0, 'Altitude At Climb Thrust Derate Deselected During Climb Below 33000 Ft')])

class TestAltitudeAtClimbThrustDerateSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtClimbThrustDerateSelection
        self.operational_combinations = [('Altitude AAL', 'Takeoff Mode (TMC) Operation', 'Airborne')]

    def test_derive(self):
        alt_aal_array = np.ma.arange(0, 10000, 2)
        alt_aal = P('Altitude AAL', array=alt_aal_array)
        tmc_array = np.ma.array([1]*3000 + [0]*2000)
        mapping = {0: '-', 1: 'Operational'}
        tmc = M(name='Takeoff Mode (TMC) Operation', array=tmc_array, values_mapping=mapping)
        airborne = buildsections('Airborne', [1000,5000])
        node = self.node_class()
        node.derive(alt_aal, tmc, airborne)
        self.assertEqual(node[0].value, 6000)
        self.assertEqual(node[0].index, 3000)


########################################
# Altitude: Gear


class TestAltitudeAtLastGearDownSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtLastGearDownSelection
        self.operational_combinations = [('Altitude AAL', 'Gear Down Selection')]

    def test_derive_basic(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(1000, 0, -100))
        gear_downs = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=4, name='Gear Down Selection'),
            KeyTimeInstance(index=8, name='Gear Down Selection')])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, gear_downs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8, value=200, name=name),
        ]))

    def test_derive_no_gear_down(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(1000, 0, -100))
        gear_downs = KTI(name='Gear Down Selection', items=[])
        node = self.node_class()
        node.derive(alt_aal, gear_downs)
        self.assertEqual(node, [])


class TestAltitudeAtGearDownSelectionWithFlapDown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearDownSelectionWithFlapDown
        self.operational_combinations = [
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever'),
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever (Synthetic)'),
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever', 'Flap Lever (Synthetic)'),
        ]

    def test_derive_basic(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=2, name='Gear Down Selection'),
            KeyTimeInstance(index=4, name='Gear Down Selection'),
            KeyTimeInstance(index=6, name='Gear Down Selection'),
            KeyTimeInstance(index=8, name='Gear Down Selection'),
        ])

        array = np.ma.concatenate((np.ones(3) * 5, np.zeros(5), np.ones(2) * 20)).astype(np.int32)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap_lever, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=2, value=200, name=name),
            KeyPointValue(index=8, value=800, name=name),
        ]))

        array = np.ma.concatenate((np.ones(3) * 5, np.ones(5), np.ones(2) * 20))
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap_lever, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=2, value=200, name=name),
            KeyPointValue(index=8, value=800, name=name),
        ]))


class TestAltitudeAtFirstGearUpSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstGearUpSelection
        self.name = 'Altitude At First Gear Up Selection'
        self.operational_combinations = [('Altitude AAL', 'Gear Up Selection')]

    def test_derive_basic(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI(name='Gear Up Selection', items=[
            KeyTimeInstance(index=4, name='Gear Up Selection'),
            KeyTimeInstance(index=8, name='Gear Up Selection')])
        node = self.node_class()
        node.derive(alt_aal, gear_downs)
        self.assertEqual(node, KPV(name=self.name, items=[
            KeyPointValue(index=4, value=400, name=self.name),
        ]))

    def test_derive_no_gear_down(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI(name='Gear Up Selection', items=[])
        node = self.node_class()
        node.derive(alt_aal, gear_downs)
        self.assertEqual(node, [])


class TestAltitudeAtGearUpSelectionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearUpSelectionDuringGoAround
        self.operational_combinations = [('Altitude AAL', 'Go Around And Climbout', 'Gear Up Selection During Go Around')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearDownSelectionWithFlapUp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearDownSelectionWithFlapUp
        self.operational_combinations = [
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever'),
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever (Synthetic)'),
            ('Altitude AAL', 'Gear Down Selection', 'Flap Lever', 'Flap Lever (Synthetic)'),
        ]

    def test_derive_basic(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=2, name='Gear Down Selection'),
            KeyTimeInstance(index=4, name='Gear Down Selection'),
            KeyTimeInstance(index=6, name='Gear Down Selection'),
            KeyTimeInstance(index=8, name='Gear Down Selection'),
        ])

        array = np.ma.concatenate((np.ones(3) * 5, np.zeros(5), np.ones(2) * 20)).astype(np.int32)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap_lever, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=4, value=400, name=name),
            KeyPointValue(index=6, value=600, name=name),
        ]))

        array = np.ma.concatenate((np.ones(3) * 5, np.ones(5), np.ones(2) * 20)).astype(np.int32)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap_lever, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=4, value=400, name=name),
            KeyPointValue(index=6, value=600, name=name),
        ]))


########################################
# Altitude: Automated Systems


class TestAltitudeAtAPEngagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtAPEngagedSelection
        self.operational_combinations = [('Altitude AAL', 'AP Engaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtAPDisengagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtAPDisengagedSelection
        self.operational_combinations = [('Altitude AAL', 'AP Disengaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtATEngagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtATEngagedSelection
        self.operational_combinations = [('Altitude AAL', 'AT Engaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtATDisengagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtATDisengagedSelection
        self.operational_combinations = [('Altitude AAL', 'AT Disengaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeWithGearDownMax
        self.operational_combinations = [
            ('Altitude AAL', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        alt_aal = P(
            name='Altitude AAL',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array(np.tile([0, 1], 5)),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(alt_aal, gear, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(index=5, value=5.0,
                                                name='Altitude With Gear Down Max'))


class TestAltitudeSTDWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeSTDWithGearDownMax
        self.operational_combinations = [
            ('Altitude STD Smoothed', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        alt_aal = P(
            name='Altitude STD Smoothed',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array(np.tile([0, 1], 5)),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(alt_aal, gear, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(index=5, value=5.0,
                                                name='Altitude STD With Gear Down Max'))


class TestAltitudeAtFirstAPEngagedAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstAPEngagedAfterLiftoff
        self.operational_combinations = [('AP Engaged', 'Altitude AAL', 'Airborne')]

    def test_derive_basic(self):

        ap = M('AP Engaged', np.ma.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), values_mapping={1:'Engaged'})
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array, alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airs = buildsection('Airborne', 2, 14)

        node = self.node_class()
        node.derive(ap=ap,alt_aal=alt_aal, airborne=airs)

        expected = KPV('Altitude At First AP Engaged After Liftoff', items=[
            KeyPointValue(name='Altitude At First AP Engaged After Liftoff', index=4.5, value=150),
        ])
        self.assertEqual(node, expected)


########################################
# Altitude: Mach


class TestAltitudeAtMachMax(unittest.TestCase, CreateKPVsAtKPVsTest):

    def setUp(self):
        self.node_class = AltitudeAtMachMax
        self.operational_combinations = [('Altitude STD Smoothed', 'Mach Max')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')



##############################################################################


class TestControlColumnStiffness(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ControlColumnStiffness
        self.operational_combinations = [('Control Column Force', 'Control Column', 'Fast')]

    def test_derive_too_few_samples(self):
        cc_disp = P('Control Column',
                    np.ma.array([0,.3,1,2,2.5,1.4,0,0]))
        cc_force = P('Control Column Force',
                     np.ma.array([0,2,4,7,8,5,2,1]))
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([100]*10)))
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff, [])

    def test_derive_max(self):
        testwave = np.ma.array((1.0 - np.cos(np.arange(0,6.3,0.1)))/2.0)
        cc_disp = P('Control Column', testwave * 10.0)
        cc_force = P('Control Column Force', testwave * 27.0)
        phase_fast = buildsection('Fast',0,63)
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff.get_first().index, 31)
        self.assertAlmostEqual(stiff.get_first().value, 2.7) # lb/deg


class TestControlColumnForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ControlColumnForceMax
        self.operational_combinations = [('Control Column Force', 'Airborne')]

    def test_derive(self):
        ccf = P(
            name='Control Column Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Airborne', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Control Column Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Control Column Force Max')]))


class TestControlWheelForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ControlWheelForceMax
        self.operational_combinations = [('Control Wheel Force', 'Airborne')]

    def test_derive(self):
        cwf = P(
            name='Control Wheel Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Airborne', 3, 9)
        node = self.node_class()
        node.derive(cwf, phase_fast)
        self.assertEqual(
            node,
            KPV('Control Wheel Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Control Wheel Force Max')]))


class TestControlColumnDualInputOppositeDirectionForceMax(unittest.TestCase):
    def setUp(self):
        self.node_class = ControlColumnDualInputOppositeDirectionForceMax

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Control Column Force At Control Wheel (Capt)', 'Control Column Force At Control Wheel (FO)',),
                                                    family=A('Family', 'ATR-72')))

    def test_derive_opposite_direction(self):
        array = np.ma.array([0, 0, 0, 0, 0, 2, 5, 20, 50, 30, 4, 2, 0, 0, 0, 0])
        cc_capt = P(name='Control Column Force At Control Wheel (Capt)', array=array)
        cc_fo = P(name='Control Column Force At Control Wheel (FO)', array=-array)
        node = self.node_class()
        node.derive(cc_capt, cc_fo, None, None)
        self.assertEqual(node[0].value, 100)
        self.assertEqual(node[0].index, 8)

    def test_derive_same_direction(self):
        array_capt = np.ma.array([0, 0, 0, 0, 0, 2, 5, 20, 50, 30, 4, 2, 0, 0, 0, 0])
        array_fo =   np.ma.array([0, 0, 0, 0, 0, 1, 4, 15, 32, 21, 5, 1, 0, 0, 0, 0])
        cc_capt = P(name='Control Column Force At Control Wheel (Capt)', array=array_capt)
        cc_fo = P(name='Control Column Force At Control Wheel (FO)', array=array_fo)
        node = self.node_class()
        node.derive(cc_capt, cc_fo, None, None)
        self.assertEqual(len(node), 0)

    def test_offset_remove(self):
        array_capt = np.ma.array([0] * 30 + [0, 0, 0, 0, 0, 2, 5, 20, 50, 30, 4, 2, 0, 0, 0, 0])
        array_fo =   np.ma.array([0] * 30 + [0, 0, 0, 0, 0, 1, 4, 15, 32, 21, 5, 1, 0, 0, 0, 0])
        array_capt += 2
        array_fo += 3
        taxi = S(items=[Section('Taxiing', slice(0,30), 0, 30)])
        turns = S('Turning On Ground')
        turns.create_sections([slice(0, 5, None),
                               slice(28, 30, None),])
        cc_capt = P(name='Control Column Force At Control Wheel (Capt)', array=array_capt)
        cc_fo = P(name='Control Column Force At Control Wheel (FO)', array=-array_fo)
        node = self.node_class()
        node.derive(cc_capt, cc_fo, taxi, turns)
        self.assertEqual(node[0].value, 82)


class TestHeightAtDistancesFromThreshold(unittest.TestCase):
    def setUp(self):
        self.node_class = HeightAtDistancesFromThreshold
        self.NAME_VALUES = {'distance': [0, 1, 2, 4]}

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Height At Distances From Threshold')
        self.assertEqual(node.NAME_FORMAT,
                         'Height At %(distance)d NM From Threshold')
        self.assertEqual(node.NAME_VALUES, self.NAME_VALUES)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 2)
        self.assertIn('Altitude AAL', opts[0])
        self.assertIn('Distance From Threshold', opts[0])

    def test_derive(self):
        alt = P('Altitude AAL', np.ma.array(np.linspace(800, 0, 55)))
        items =[]
        indices = [40, 30, 20, 10]
        for d, i in zip(self.NAME_VALUES['distance'], indices):
            items.append(KeyTimeInstance(index=i,
                                         name='%d NM From Threshold' % d))
        dist_ktis = DistanceFromThreshold(name='Distance From Threshold',
                                          items=items)

        node = self.node_class()
        node.derive(alt, dist_ktis)

        self.assertEqual(len(node), 4)

        self.assertEqual(node[0].name, 'Height At 0 NM From Threshold')
        self.assertEqual(node[0].index, 40)
        self.assertAlmostEqual(node[0].value, 207.41, places=2)

        self.assertEqual(node[1].name, 'Height At 1 NM From Threshold')
        self.assertEqual(node[1].index, 30)
        self.assertAlmostEqual(node[1].value, 355.56, places=2)

        self.assertEqual(node[2].name, 'Height At 2 NM From Threshold')
        self.assertEqual(node[2].index, 20)
        self.assertAlmostEqual(node[2].value, 503.70, places=2)

        self.assertEqual(node[3].name, 'Height At 4 NM From Threshold')
        self.assertEqual(node[3].index, 10)
        self.assertAlmostEqual(node[3].value, 651.85, places=2)


class TestHeightAtOffsetILSTurn(unittest.TestCase):
    def setUp(self):
        self.node_class = HeightAtOffsetILSTurn

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Height At Offset ILS Turn')
        self.assertEqual(node.units, ut.FT)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 2)
        self.assertIn('Altitude AAL', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive(self):
        alt = P('Altitude AAL', np.ma.array(np.linspace(800, 0, 55)))
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(10, 40),
                                       offset_ils=True,
                                       loc_est=slice(20, 30))])

        node = self.node_class()
        node.derive(alt, apps)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 30)
        self.assertAlmostEqual(node[0].value, 355.56, places=2)


class TestHeightAtRunwayChange(unittest.TestCase):
    def setUp(self):
        self.node_class = HeightAtRunwayChange

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Height At Runway Change')
        self.assertEqual(node.units, ut.FT)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 2)
        self.assertIn('Altitude AAL', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive(self):
        alt = P('Altitude AAL', np.ma.array(np.linspace(800, 0, 55)))
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(10, 40),
                                       runway_change=True,
                                       loc_est=slice(20, 30))])

        node = self.node_class()
        node.derive(alt, apps)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 30)
        self.assertAlmostEqual(node[0].value, 355.56, places=2)


class TestHeightSelectedOnApproachMin(unittest.TestCase):
    def setUp(self):
        self.node_class = HeightSelectedOnApproachMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Altitude Selected (MCP)', 'Approach Information')])

    def test_basic(self):
        alt_mcp = P(name='Altitude Selected (MCP)',
                    array=np.ma.array([1000,150,250]))
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING',
                                           slice(0, 3),
                                           airport={u'elevation': 35,
                                                    u'name': u'Test'}
                                           )])
        node = self.node_class()
        node.derive(alt_mcp, apps)
        self.assertEqual(node[0].value, 115) # 150-35

    def test_not_negative(self):
        alt_mcp = P(name='Altitude Selected (MCP)',
                    array=np.ma.array([1000,150,250]))
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING',
                                           slice(0, 3),
                                           airport={u'elevation': 350,
                                                    u'name': u'Test'}
                                           )])
        node = self.node_class()
        node.derive(alt_mcp, apps)
        self.assertEqual(node[0].value, 0) # 150-350 = -200


##############################################################################
# Baro Correction


class TestQNHDifferenceDuringApproach(unittest.TestCase):
    def setUp(self):
        self.node_class = QNHDifferenceDuringApproach

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, 'mb')
        self.assertEqual(node.name,
                         'QNH Difference During Approach')

    def test_derive_single_app(self):
        alt_qnh = P(name='Altitude QNH',
                    array=np.ma.array([210, 180, 160, 140, 120, 100]))
        alt_viz = P('Altitude Visualization with Ground Offset',
                    array=np.ma.array([270, 240, 220, 200, 180, 160]))
        alt_all = P('Altitude AAL',
                    array=np.ma.array([110, 80, 60, 40, 20, 0]))
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(0, 6))])

        node = self.node_class()
        node.derive(alt_qnh, alt_viz, alt_all, apps)
        # 60 ft below is roughly 2 mBar difference
        self.assertAlmostEqual(node[0].value, -2, delta=0.2)
        self.assertAlmostEqual(node[0].index, 1/3., delta=0.01)

    def test_derive_multiple_app(self):
        alt_qnh = P(name='Altitude QNH',
                    array=np.ma.array([
                        400, 380, 350, 320, 330, 350,    # Go around
                        275, 245, 225, 205, 185, 165]))  #  Landing
        alt_viz = P('Altitude Visualization with Ground Offset',
                    array=np.ma.array([
                        460, 440, 410, 380, 390, 410,    # Go around
                        270, 240, 220, 200, 180, 160]))  # Landing
        alt_aal = P('Altitude AAL',
                    array=np.ma.array([
                        300, 280, 250, 220, 230, 250,  # Go around
                        110, 80, 60, 40, 20, 0]))      # Landing
        apps = App('Approach Information',
                   items=[ApproachItem('GO-AROUND', slice(0, 5)),
                          ApproachItem('LANDING', slice(6, 12))])

        node = self.node_class()
        node.derive(alt_qnh, alt_viz, alt_aal, apps)
        self.assertEqual(len(node), 2)
        # 60 ft below is roughly 2 mBar difference
        self.assertAlmostEqual(node[0].value, -2, delta=0.2)
        self.assertEqual(node[0].index, 3)
        # Second approach had only 5 ft diff. Almost 0 mBar difference
        self.assertAlmostEqual(node[1].value, 0, delta=0.2)
        self.assertAlmostEqual(node[1].index, 6 + 1/3., delta=0.01)


class TestBaroCorrectionMinus1013Above20000FtMax(unittest.TestCase):
    def setUp(self):
        self.node_class = BaroCorrectionMinus1013Above20000FtDuringLevelFlightMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, 'mb')
        self.assertEqual(node.name,
                         'Baro Correction Minus 1013 Above 20000 Ft During Level Flight Max')

    def test_can_operate(self):
        ops = set(self.node_class.get_operational_combinations())
        expected = {('Baro Correction', 'Altitude STD', 'Level Flight'),
                    ('Altitude STD', 'Level Flight')}
        self.assertEqual(ops, expected)

    def test_baro_correction_below_std(self):
        node = self.node_class()
        # Climb to 30,000ft, cruise and descend
        alt_std = P('Altitude STD',
                    array=np.ma.concatenate([
                        np.linspace(0, 30000, num=31),
                        np.ma.ones(28) * 30000,
                        np.linspace(30000, 0, num=31)
                    ]))
        baro = P('Baro Correction', array=np.ma.ones(90) * 1010)
        level = buildsection('Level Flight', 30, 59)
        node.derive(baro, alt_std, level)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[0].index, 30)

    def test_baro_correction_no_difference(self):
        node = self.node_class()
        # Climb to 30,000ft, cruise and descend
        alt_std = P('Altitude STD',
                    array=np.ma.concatenate([
                        np.linspace(0, 30000, num=31),
                        np.ma.ones(28) * 30000,
                        np.linspace(30000, 0, num=31)
                    ]))
        baro = P('Baro Correction', array=np.ma.concatenate([
            np.ma.ones(10) * 999,
            np.ma.ones(70) * 1013.8,
            np.ma.ones(10) * 1021
        ]))
        level = buildsection('Level Flight', 30, 59)
        node.derive(baro, alt_std, level)

        self.assertEqual(len(node), 0)

    def test_2_level_sections(self):
        node = self.node_class()
        # Climb to 18,000ft, cruise, climb to 30,000ft, cruise and descend
        alt_std = P('Altitude STD',
                    array=np.ma.concatenate([
                        np.linspace(0, 18000, num=19),
                        np.ma.ones(13) * 18000,
                        np.linspace(18000, 30000, num=13),
                        np.ma.ones(14) * 30000,
                        np.linspace(30000, 0, num=31)
                    ]))
        baro = P('Baro Correction', array=np.ma.ones(90) * 1010)
        level = buildsections('Level Flight', [18, 33], [45, 59])
        node.derive(baro, alt_std, level)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[0].index, 45)

    def test_change_qnh_in_descent(self):
        node = self.node_class()
        alt_std = P('Altitude STD',
                    array=np.ma.concatenate([
                        np.linspace(0, 30000, num=31),
                        np.ma.ones(28) * 30000,
                        np.linspace(30000, 0, num=31)
                    ]))
        baro = P('Baro Correction',
                 array=np.ma.concatenate([
                     np.ma.ones(15) * 1010,
                     np.ma.ones(50) * 1013,
                     np.ma.ones(25) * 1005,
                 ]))
        level = buildsection('Level Flight', 30, 59)
        node.derive(baro, alt_std, level)

        self.assertEqual(len(node), 0)

    def test_alt_std_only(self):
        node = self.node_class()
        # Climb to 29,700ft, cruise, climb to 30,000ft, cruise and descend
        alt_std = P('Altitude STD',
                    array=np.ma.concatenate([
                        np.linspace(0, 29700, num=31),
                        np.ma.ones(8) * 29700,
                        np.linspace(29700, 30000, num=5),
                        np.ma.ones(15) * 30000,
                        np.linspace(30000, 0, num=31)
                    ]))
        level = buildsections('Level Flight', [30, 38], [43, 59])
        node.derive(None, alt_std, level)

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 300.0/27.3, delta = 0.1)
        self.assertEqual(node[0].index, 30)


class TestBaroDifferenceDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = BaroDifferenceDuration

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, 's')
        self.assertEqual(node.name,
                         'Baro Difference Duration')

    def test_differ_for_20_sec(self):
        baro_diff = buildsection('Baro Difference', 10, 15)
        baro_diff.hz = 1./4.

        node = self.node_class()
        node.get_derived((baro_diff,))

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 5 * 4)  # 5 samples at 1/4 Hz
        self.assertEqual(node[0].index, 10)

    def test_no_difference(self):
        baro_diff = S('Baro Difference')
        baro_diff.hz = 1./4.

        node = self.node_class()
        node.get_derived((baro_diff,))

        self.assertEqual(len(node), 0)

    def test_multiple_differences(self):
        baro_diff = buildsections('Baro Difference', [0, 3], [10, 15])
        baro_diff.hz = 1./4.

        node = self.node_class()
        node.get_derived((baro_diff,))

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 3 * 4)  # 3 sample at 1/4 Hz
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[1].value, 5 * 4)  # 5 samples at 1/4 Hz
        self.assertEqual(node[1].index, 10)


##############################################################################
# Heading


class TestHeadingDuringTakeoff(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingDuringTakeoff

    def test_can_operate(self):
        can_op = self.node_class.can_operate
        self.assertTrue(can_op(('Heading Continuous', 'Transition Hover To Flight', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(can_op(('Heading Continuous', 'Takeoff Roll Or Rejected Takeoff'), ac_type=aeroplane))

    def test_derive_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        toff = buildsection('Takeoff', 2,5)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=7.5,
                                  name='Heading During Takeoff')]
        self.assertEqual(kpv, expected)

    def test_derive_modulus(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,5)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5,
                                  name='Heading During Takeoff')]
        self.assertEqual(kpv, expected)


class TestHeadingTrueDuringTakeoff(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingTrueDuringTakeoff

    def test_can_operate(self):
        can_op = self.node_class.can_operate
        self.assertTrue(can_op(('Heading True Continuous', 'Transition Hover To Flight', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(can_op(('Heading True Continuous', 'Takeoff Roll Or Rejected Takeoff'), ac_type=aeroplane))

    def test_derive_basic(self):
        head = P('Heading True Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        toff = buildsection('Takeoff', 2,5)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=7.5,
                                  name='Heading True During Takeoff')]
        self.assertEqual(kpv, expected)

    def test_derive_modulus(self):
        head = P('Heading True Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,5)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5,
                                  name='Heading True During Takeoff')]
        self.assertEqual(kpv, expected)


class TestHeadingDuringLanding(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingDuringLanding

    def test_can_operate(self):
        can_op = self.node_class.can_operate
        self.assertTrue(can_op(('Heading Continuous', 'Transition Flight To Hover', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(can_op(('Heading Continuous', 'Landing Roll'), ac_type=aeroplane))

    def test_derive_fw(self):
        head = P('Heading Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        ac_type=A(name='Aircraft Type', value='aircraft')
        landing = buildsection('Landing', 5, 14)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, landing, ac_type, None)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)

    def test_derive_helo(self):
        head = P('Heading Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        ac_type=A(name='Aircraft Type', value='helicopter')
        landing = buildsection('Transition Flight To Hover', 5, 14)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, None, ac_type, landing)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)


class TestHeadingTrueDuringLanding(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingTrueDuringLanding

    def test_can_operate(self):
        can_op = self.node_class.can_operate
        self.assertTrue(can_op(('Heading True Continuous', 'Transition Flight To Hover', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(can_op(('Heading True Continuous', 'Landing Roll'), ac_type=aeroplane))

    def test_derive_fw(self):
        # Duplicate of TestHeadingDuringLanding.test_derive_basic.
        head = P('Heading True Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        ac_type=A(name='Aircraft Type', value='aircraft')
        landing = buildsection('Landing', 5, 14)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, landing, ac_type, None)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading True During Landing')]
        self.assertEqual(kpv, expected)

    def test_derive_helo(self):
        # Duplicate of TestHeadingDuringLanding.test_derive_basic.
        head = P('Heading True Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        ac_type=A(name='Aircraft Type', value='helicopter')
        landing = buildsection('Transition Flight To Hover', 5, 14)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, None, ac_type, landing)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading True During Landing')]
        self.assertEqual(kpv, expected)

class TestHeadingAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = HeadingAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Heading Continuous', 'Lowest Altitude During Approach')]

    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        # derive() uses par1 % 360.0, so the par1 needs to be compatible with %
        # operator
        mock1.array = 0
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingChange(unittest.TestCase):
    def setUp(self):
        self.node_class = HeadingChange
        self.operational_combinations = [('Heading Continuous',
                                          'Turning In Air')]

    def test_derive_basic(self):
        head = P('Heading Continuous',
                 np.ma.concatenate((np.zeros(10), np.ones(10) * 280.0)))
        landing = buildsection('Turning In Air', 5, 14)
        kpv = HeadingChange()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=14, value=280.0,
                                  name='Heading Change')]
        self.assertEqual(kpv, expected)

    def test_derive_left(self):
        head = P('Heading Continuous',
                 np.ma.concatenate((np.zeros(10), np.ones(10) * -300.0)))
        landing = buildsection('Turning In Air', 5, 14)
        kpv = HeadingChange()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=14, value=-300.0,
                                  name='Heading Change')]
        self.assertEqual(kpv, expected)

    def test_derive_small_angle(self):
        head = P('Heading Continuous',
                 np.ma.concatenate((np.zeros(10), np.ones(10) * 269.0)))
        landing = buildsection('Turning In Air', 5, 14)
        kpv = HeadingChange()
        kpv.derive(head, landing)
        self.assertEqual(len(kpv), 0)


class TestHeadingRateWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = HeadingRateWhileAirborneMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Heading Rate', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 50)
        heading_rate = P(
            name='Heading Rate',
            array=-x*np.sin(x),
        )
        name = 'Airborne'
        section = Section(name, slice(0, 150), 0, 150)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(heading_rate, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 39)
        self.assertAlmostEqual(node[0].value, -7.915, places=3)


##############################################################################
# Flight Path Angle

class TestApproachFlightPathAngle1500To1000FtMax(unittest.TestCase,
                                                 CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle1500To1000FtMax
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1500, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestApproachFlightPathAngle1500To1000FtMin(unittest.TestCase,
                                                 CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle1500To1000FtMin
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1500, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestApproachFlightPathAngle1000To500FtMax(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle1000To500FtMax
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestApproachFlightPathAngle1000To500FtMin(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle1000To500FtMin
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestApproachFlightPathAngle500To200FtMax(unittest.TestCase,
                                               CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle500To200FtMax
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 200), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestApproachFlightPathAngle500To200FtMin(unittest.TestCase,
                                               CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle500To200FtMin
        self.operational_combinations = [('Approach Flight Path Angle',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 200), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# ILS


class TestILSFrequencyDuringApproach(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSFrequencyDuringApproach
        self.operational_combinations = [('Approach Information',)]

    @unittest.skip('Using Approach Information makes this trivial')
    def test_derive_basic(self):
        pass


class TestILSGlideslopeDeviation1500To1000FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation1500To1000FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
            ), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate((
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
            ), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Glideslope',
            'ILS Glideslope Established',
            'Altitude AGL',
            'Descending',
            ), ac_type=helicopter))

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 31)
        self.assertAlmostEqual(kpv[0].value, 1.99913515027)

    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 16)
        self.assertAlmostEqual(kpv[0].value, -1.1995736)


class TestILSGlideslope10SecBeforeEstablished(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSGlideslope10SecBeforeEstablished


    def test_derive_positive_dot(self):

        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(1.0 - np.cos(np.arange(0, 6.3, 0.1))),
        )

        ils_established = buildsection('ILS Glideslope Established', 30, 60)

        kpv = ILSGlideslope10SecBeforeEstablished()
        kpv.derive(ils_glideslope, ils_established)

        self.assertEqual(len(kpv),1)
        self.assertGreater(kpv[0].value, 0)
        self.assertEqual(kpv[0].index, 20)


    def test_derive_multiple_approaches(self):

        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(1.0 - np.cos(np.arange(0, 6.3, 0.1))),
        )

        ils_glideslope.array[20] = -ils_glideslope.array[20]
        ils_established = buildsections('ILS Glideslope Established', [30,50], [40, 60])

        kpv = ILSGlideslope10SecBeforeEstablished()
        kpv.derive(ils_glideslope, ils_established)

        self.assertEqual(len(kpv),2)
        self.assertLess(kpv[0].value, 0)
        self.assertGreater(kpv[1].value, 0)
        self.assertEqual(kpv[0].index, 20)
        self.assertEqual(kpv[1].index, 30)


    def test_derive_negative_dot(self):

        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(1.0 - np.cos(np.arange(0, 6.3, 0.1))),
        )

        ils_glideslope.array = -ils_glideslope.array
        ils_established = buildsection('ILS Glideslope Established', 30, 60)

        kpv = ILSGlideslope10SecBeforeEstablished()
        kpv.derive(ils_glideslope, ils_established)

        self.assertEqual(len(kpv),1)
        self.assertLess(kpv[0].value, 0)
        self.assertEqual(kpv[0].index, 20)


    def test_derive_masked_value(self):

        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(1.0 - np.cos(np.arange(0, 6.3, 0.1))),
        )

        ils_glideslope.array[40] = np.ma.masked
        ils_established = buildsection('ILS Glideslope Established', 50, 60)

        kpv = ILSGlideslope10SecBeforeEstablished()
        kpv.derive(ils_glideslope, ils_established)

        self.assertEqual(len(kpv),0)


class TestILSGlideslopeDeviation1000To500FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation1000To500FtMax
        self.operational_combinations = [(
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
        )]

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation1000To500FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 36)
        self.assertAlmostEqual(kpv[0].value, 1.89675842)

    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation1000To500FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 47)
        self.assertAlmostEqual(kpv[0].value, 0.79992326)


class TestILSGlideslopeDeviation500To200FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation500To200FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate((
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
            ), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'ILS Glideslope',
            'ILS Glideslope Established',
            'Altitude AGL',
            'Descending',
            ), ac_type=helicopter))

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation500To200FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 56)
        self.assertAlmostEqual(kpv[0].value, 0.22443412)

    # FIXME: Need to amend the test data as it produces no key point value for
    #        the 500-200ft altitude range. Originally this was not a problem
    #        before we split the 1000-250ft range in two.
    @unittest.skip('Test does not work... Need to amend test data.')
    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation500To200FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 0)          # FIXME
        self.assertAlmostEqual(kpv[0].value, 0.0)  # FIXME


class TestILSGlideslopeNotEstablishedHighestAltitude1000To200Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSGlideslopeNotEstablishedHighestAltitude1000To200Ft

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AAL For Flight Phases',
            'ILS Glideslope',
            'ILS Localizer'
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AGL',
            'ILS Glideslope',
            'ILS Localizer'
            ), ac_type=helicopter))

    def test_derive_ILS_established_during_one_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_ILS_established_two_approaches(self):

        alt_aal = P('Altitude AAL', np.tile(np.ma.arange(4000, 0, -10),2))
        ils_gs = P('ILS GLideslope', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ils_loc = P('ILS Localizer', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)
        self.assertEqual(node[1].index, 780)
        self.assertEqual(node[1].value, 200)

    def test_derive_ILS_not_established(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(1.0, 2.0, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)

    def test_derive_ILS_not_established_at_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(1.0, 2.0, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)

    def test_derive_helicopter_ILS_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_helicopter_ILS_not_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(1.5, 2.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)


class TestILSGlideslopeNotEstablishedHighestAltitude500To200Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSGlideslopeNotEstablishedHighestAltitude500To200Ft

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AAL For Flight Phases',
            'ILS Glideslope',
            'ILS Localizer'
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AGL',
            'ILS Glideslope',
            'ILS Localizer'
            ), ac_type=helicopter))

    def test_derive_ILS_established_during_one_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_ILS_established_two_approaches(self):

        alt_aal = P('Altitude AAL', np.tile(np.ma.arange(4000, 0, -10),2))
        ils_gs = P('ILS GLideslope', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ils_loc = P('ILS Localizer', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)
        self.assertEqual(node[1].index, 780)
        self.assertEqual(node[1].value, 200)

    def test_derive_ILS_not_established(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(1.0, 2.0, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 500)
        self.assertEqual(node[0].index, 351)

    def test_derive_helicopter_ILS_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_helicopter_ILS_not_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_gs = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc = P('ILS Localizer', np.arange(1.0, 2.0, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_gs, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 500)
        self.assertEqual(node[0].index, 351)


class TestILSLocalizerNotEstablishedHighestAltitude1000To200Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSLocalizerNotEstablishedHighestAltitude1000To200Ft

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AAL For Flight Phases'
            'ILS Localizer'
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AGL',
            'ILS Localizer'
            ), ac_type=helicopter))

    def test_derive_ILS_established_during_one_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_ILS_established_two_approaches(self):

        alt_aal = P('Altitude AAL', np.tile(np.ma.arange(4000, 0, -10),2))
        ils_loc = P('ILS GLideslope', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_loc)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)
        self.assertEqual(node[1].index, 780)
        self.assertEqual(node[1].value, 200)

    def test_derive_ILS_not_established(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(1.0, 2.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)

    def test_derive_ILS_not_established_at_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc.array[200:400] = np.arange(2.0, 3.0)
        ac_type = A('Aircraft Type', 'aeroplane')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)

    def test_derive_helicopter_ILS_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_helicopter_ILS_not_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc.array[200:400] = np.arange(2.0, 3.0)
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1000)
        self.assertEqual(node[0].index, 301)


class TestILSLocalizerNotEstablishedHighestAltitude500To200Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSLocalizerNotEstablishedHighestAltitude500To200Ft

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AAL For Flight Phases'
            'ILS Localizer'
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'Altitude AGL',
            'ILS Localizer'
            ), ac_type=helicopter))

    def test_derive_ILS_established_during_one_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_ILS_established_two_approaches(self):

        alt_aal = P('Altitude AAL', np.tile(np.ma.arange(4000, 0, -10),2))
        ils_loc = P('ILS GLideslope', np.repeat(np.arange(-0.5, 0.5, 0.0025),2))
        ac_type = A('Aircraft Type', 'aeroplane')

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, None, ils_loc)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)
        self.assertEqual(node[1].index, 780)
        self.assertEqual(node[1].value, 200)

    def test_derive_ILS_not_established(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(1.0, 2.5, 0.0025))
        ac_type = A('Aircraft Type', 'aeroplane')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 500)
        self.assertEqual(node[0].index, 351)

    def test_derive_ILS_not_established_at_approach(self):

        alt_aal = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc.array[200:400] = np.arange(2.0, 3.0)
        ac_type = A('Aircraft Type', 'aeroplane')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(alt_aal, None,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 500)
        self.assertEqual(node[0].index, 351)

    def test_derive_helicopter_ILS_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 380)
        self.assertEqual(node[0].value, 200)

    def test_derive_helicopter_ILS_not_established(self):

        alt_agl = P('Altitude AAL', np.ma.arange(4000, 0, -10))
        ils_loc = P('ILS GLideslope', np.arange(-0.5, 0.5, 0.0025))
        ils_loc.array[200:400] = np.arange(2.0, 3.0)
        ac_type = A('Aircraft Type', 'helicopter')
        approach = S(items=[Section('Approach', slice(200, 400), 200, 400)])

        node = self.node_class()
        node.derive(None, alt_agl,  ac_type, approach, ils_loc)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 500)
        self.assertEqual(node[0].index, 351)


class TestILSLocalizerDeviation1500To1000FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation1500To1000FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'ILS Localizer Established',
            'Altitude AGL',
            'Descending',
            ), ac_type=helicopter))

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        self.assertAlmostEqual(kpv[0].value, 4.7)
        self.assertAlmostEqual(kpv[1].value, 10.9)


class TestILSLocalizerDeviation1000To500FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation1000To500FtMax
        self.operational_combinations = [(
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
        )]

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'ILS Localizer Established',
            'Altitude AGL',
            'Descending',
            ), ac_type=helicopter))

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation1000To500FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 52)
        self.assertEqual(kpv[1].index, 114)
        self.assertAlmostEqual(kpv[0].value, 5.2)
        self.assertAlmostEqual(kpv[1].value, 11.4)


class TestILSLocalizerDeviation500To200FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation500To200FtMax
        self.operational_combinations = [(
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
        )]

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate((
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
            ), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate((
            'ILS Localizer',
            'ILS Localizer Established',
            'Altitude AGL',
            'Descending',
            ), ac_type=helicopter))

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation500To200FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 56)
        self.assertAlmostEqual(kpv[0].value, 5.6)


class TestILSLocalizerDeviationAtTouchdown(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviationAtTouchdown
        self.operational_combinations = [(
            'ILS Localizer',
            'ILS Localizer Established',
            'Touchdown',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive_basic(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestIANGlidepathDeviationMax(unittest.TestCase):

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('IAN Glidepath', 'Altitude AAL For Flight Phases', 'IAN Glidepath Established')]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANGlidepathDeviationMax
        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.ian = P(name='IAN Glidepath', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 2.45, 0, 0, 0, 0, 0], dtype=np.float,))
        self.established = buildsection('IAN Glidepath Established', 5, 12)

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.established)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 2.45)
        self.assertAlmostEqual(kpv[0].name, 'IAN Glidepath Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, S('IAN Glidepath Established'))
        self.assertEqual(len(kpv), 0)

    def test_derive_with_real_data(self):
        ian_array = load_compressed(os.path.join(test_data_path, 'ian_established-ian_glidepath.npz'))
        ian_glidepath = P('IAN Glidepath', ian_array)
        aal_array = load_compressed(os.path.join(test_data_path, 'ian_established-alt_aal.npz'))
        alt_aal = P('Altitude AAL For Flight Phases', aal_array)
        established = buildsection('IAN Glidepath Established', 30346, 30419)

        node = self.node_class()
        node.derive(ian_glidepath, alt_aal, established)

        # established phase starts below 1000ft
        self.assertEqual(len(node), 2)

        self.assertEqual(node[0].index, 30389)
        self.assertAlmostEqual(node[0].value, 0.74, delta=0.01)
        self.assertEqual(node[0].name, 'IAN Glidepath Deviation 1000 To 500 Ft Max')

        self.assertEqual(node[1].index, 30418)
        self.assertAlmostEqual(node[1].value, 1.74, delta=0.01) # check
        self.assertEqual(node[1].name, 'IAN Glidepath Deviation 500 To 200 Ft Max')


class TestIANFinalApproachCourseDeviationMax(unittest.TestCase):

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('IAN Final Approach Course', 'Altitude AAL For Flight Phases', 'IAN Final Approach Course Established'),]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANFinalApproachCourseDeviationMax
        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.ian = P(name='IAN Final Approach Course', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 2.45, 0, 0, 0, 0, 0], dtype=np.float,))
        self.established = buildsection('IAN Final Approach Course Established', 5, 12)

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.established)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 2.45)
        self.assertAlmostEqual(kpv[0].name, 'IAN Final Approach Course Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, S('IAN Final Approach Course Established'))
        self.assertEqual(len(kpv), 0)

    def test_derive_with_real_data(self):
        ian_array = load_compressed(os.path.join(test_data_path, 'ian_established-ian_app_course.npz'))
        ian_app_corse = P('IAN Final Approach Course', ian_array)
        aal_array = load_compressed(os.path.join(test_data_path, 'ian_established-alt_aal.npz'))
        alt_aal = P('Altitude AAL For Flight Phases', aal_array)
        established = buildsection('IAN Final Approach Course Established', 30238, 30537)

        node = self.node_class()
        node.derive(ian_app_corse, alt_aal, established)

        self.assertEqual(len(node), 3)

        self.assertEqual(node[0].index, 30341)
        self.assertAlmostEqual(node[0].value, 0.09, delta=0.01)
        self.assertEqual(node[0].name, 'IAN Final Approach Course Deviation 1500 To 1000 Ft Max')

        self.assertEqual(node[1].index, 30362)
        self.assertAlmostEqual(node[1].value, 1.83, delta=0.01)
        self.assertEqual(node[1].name, 'IAN Final Approach Course Deviation 1000 To 500 Ft Max')

        self.assertEqual(node[2].index, 30396)
        self.assertAlmostEqual(node[2].value, 1.09, delta=0.01) # Check
        self.assertEqual(node[2].name, 'IAN Final Approach Course Deviation 500 To 200 Ft Max')

##############################################################################
# Mach


########################################
# Mach: General


class TestMachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachMax
        self.operational_combinations = [('Mach', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestMachDuringCruiseAvg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachDuringCruiseAvg
        self.operational_combinations = [('Mach', 'Cruise')]

    def test_derive_basic(self):
        mach_array = np.ma.concatenate([np.arange(0, 1, 0.1),
                                        np.arange(1, 0, -0.1)])
        mach = P('Mach', array=mach_array)
        cruise = buildsection('Cruise', 5, 10)
        node = self.node_class()
        node.derive(mach, cruise)
        self.assertEqual(node[0].index, 7)
        self.assertAlmostEqual(node[0].value,0.7)
        self.assertEqual(node[0].name, 'Mach During Cruise Avg')


class TestMachAboveFL200Max(unittest.TestCase):
    def can_operate(self):
        opts = MachAboveFL200Max.get_operational_combinations()
        self.assertEqual(opts, [('Mach', 'Altitude STD Smoothed')])

    def test_derive(self):
        mach = P('Mach', array=[0.85, 0.86, 0.84, 0.82, 0.84, 0.85, 0.87, 0.90, 0.91])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = MachAboveFL200Max()
        node.derive(mach, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 0.85)

class TestMachAboveFL200Min(unittest.TestCase):
    def can_operate(self):
        opts = MachAboveFL200Min.get_operational_combinations()
        self.assertEqual(opts, [('Mach', 'Altitude STD Smoothed')])

    def test_derive(self):
        mach = P('Mach', array=[0.85, 0.86, 0.84, 0.82, 0.84, 0.85, 0.87, 0.90, 0.91])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = MachAboveFL200Min()
        node.derive(mach, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 0.84)

########################################
# Mach: Flap


class TestMachWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachWithFlapMax
        self.operational_combinations = [
            ('Flap Lever', 'Mach', 'Fast'),
            ('Flap Lever (Synthetic)', 'Mach', 'Fast'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Mach', 'Fast'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Mach: Landing Gear


class TestMachWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachWithGearDownMax
        self.operational_combinations = [('Mach', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        mach = P(
            name='Mach',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(mach, gear, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(index=5, value=5.0, name='Mach With Gear Down Max'))


class TestMachWhileGearRetractingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachWhileGearRetractingMax
        self.operational_combinations = [('Mach', 'Gear Retracting')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestMachWhileGearExtendingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachWhileGearExtendingMax
        self.operational_combinations = [('Mach', 'Gear Extending')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
########################################


class TestAltitudeFirstStableDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeFirstStableDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_stable_with_one_approach(self):
        firststable = AltitudeFirstStableDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,9,9,3,2,9,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,500,400,300]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 1.5)
        self.assertEqual(firststable[0].value, 850)

    def test_derive_stable_with_one_approach_all_unstable(self):
        firststable = AltitudeFirstStableDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,3,3,3,2,3,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,400,140,0]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 7.5)
        self.assertEqual(firststable[0].value, 0)

    def test_derive_two_approaches(self):
        # two approaches
        firststable = AltitudeFirstStableDuringLastApproach()
        #                                            stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9],
                                                  mask=[0,0,0,0,  1,0,0,   0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100]))
        firststable.derive(stable, alt2app)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 7.5)
        self.assertEqual(firststable[0].value, 250)


class TestAltitudeFirstStableDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeFirstStableDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches_keeps_only_first(self):
        # two approaches
        firststable = AltitudeFirstStableDuringApproachBeforeGoAround()
        #                                            stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9],
                                                  mask=[0,0,0,0,  1,0,0,   0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100]))
        firststable.derive(stable, alt2app)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 1.5)
        self.assertEqual(firststable[0].value, 850)

    def test_derive_stable_with_one_approach_is_ignored(self):
        firststable = AltitudeFirstStableDuringApproachBeforeGoAround()
        stable = StableApproach(array=np.ma.array([1,4,9,9,3,2,9,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,500,400,300]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 0)


class TestAltitudeLastUnstableDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeLastUnstableDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches_uses_last_one(self):
        # two approaches
        lastunstable = AltitudeLastUnstableDuringLastApproach()
        #                                                 stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9,1,1],
                                                  mask=[0,0,0,0,  1,0,0,   0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,20,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)  # will always land with AAL of 0

    def test_never_stable_stores_a_value(self):
        # if we were never stable, ensure we record a value at landing (0 feet)
        lastunstable = AltitudeLastUnstableDuringLastApproach()
        # not stable for either approach
        stable = StableApproach(array=np.ma.array([1,4,4,4,  3,2,2,2,2,2,1,1],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,50,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)


class TestAltitudeLastUnstableDuringLastApproachExcludingEngThrust(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeLastUnstableDuringLastApproachExcludingEngThrust.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach Excluding Eng Thrust',
                                'Altitude AAL')])

    def test_derive_two_approaches_uses_last_one(self):
        # two approaches
        lastunstable = AltitudeLastUnstableDuringLastApproachExcludingEngThrust()
        #                                                 stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9,1,1],
                                                  mask=[0,0,0,0,  1,0,0,   0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,20,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)  # will always land with AAL of 0

    def test_never_stable_stores_a_value(self):
        # if we were never stable, ensure we record a value at landing (0 feet)
        lastunstable = AltitudeLastUnstableDuringLastApproachExcludingEngThrust()
        # not stable for either approach
        stable = StableApproach(array=np.ma.array([1,4,4,4,  3,2,2,2,2,2,1,1],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,50,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)


class TestAltitudeLastUnstableDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeLastUnstableDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches(self):
        # two approaches
        lastunstable = AltitudeLastUnstableDuringApproachBeforeGoAround()
        #                                                 stable tooshort stable  last
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,2,9,9,1,1, 1,3,9,9,9],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0, 1,0,0,0,0]))
        alt2app = P(array=np.ma.array([1500,1400,1200,1000,
                                       900,800,700,600,500,400,300,200,
                                       100,50,20,0,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 2)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 1.5)
        self.assertEqual(lastunstable[0].value, 1300)
        self.assertEqual(lastunstable[1].index, 11.5)
        self.assertEqual(lastunstable[1].value, 0)  # was not stable prior to go around

    def test_never_stable_reads_0(self):
        lastunstable = AltitudeLastUnstableDuringApproachBeforeGoAround()
        # not stable for either approach
        stable = StableApproach(array=np.ma.array([1,4,4,4,  3,2,2,2,2,2,1,1],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,50,20]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 3.5)
        self.assertEqual(lastunstable[0].value, 0)


class TestLastUnstableStateDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = LastUnstableStateDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach',)])

    def test_derive(self):
        state = LastUnstableStateDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,4,2,9,9,9,9],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        state.derive(stable)
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].index, 7.5)
        self.assertEqual(state[0].value, 2)

    @unittest.skip('This is so unlikely that its deemed unrealistic')
    def test_last_unstable_state_if_always_stable(self):
        # pas possible
        pass


class TestLastUnstableStateDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = LastUnstableStateDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach',)])

    def test_derive(self):
        state = LastUnstableStateDuringApproachBeforeGoAround()
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,4,2,9,9,9,9],
                                                  mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        state.derive(stable)
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].index, 1.5)
        self.assertEqual(state[0].value, 4)


class TestPercentApproachStableBelow(unittest.TestCase):
    def test_can_operate(self):
        ops = PercentApproachStable.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches(self):
        percent_stable = PercentApproachStable()
        stable = StableApproach(array=np.ma.array([1,4,9,9,9, 3, 2,9,2,9,9,1,1],
                                                  mask=[0,0,0,0,0, 1, 0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1100,1000,900,800,700,
                                       600,
                                       600,650,200,100,50,20,1]))
        percent_stable.derive(stable, alt2app)
        # both approaches below
        self.assertEqual(len(percent_stable), 3)

        # First approach reaches only 1000 feet barrier (does not create 500ft)
        self.assertEqual(percent_stable[0].name,
                         "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[0].index, 2)
        self.assertEqual(percent_stable[0].value, 75)  #3/4 == 75% - 4 samples below 1000ft

        # Last approach is below 1000 and 500 feet
        self.assertEqual(percent_stable[1].name,
                         "Percent Approach Stable Below 1000 Ft During Last Approach")
        self.assertEqual(percent_stable[1].index, 7)
        self.assertEqual(percent_stable[1].value, (3/7.0)*100)  #3/7

        self.assertEqual(percent_stable[2].name,
                         "Percent Approach Stable Below 500 Ft During Last Approach")
        self.assertEqual(percent_stable[2].index, 9)
        self.assertEqual(percent_stable[2].value, 40)  #2/5 == 40%

    def test_derive_three_approaches(self):
        # three approaches
        percent_stable = PercentApproachStable()
        stable = StableApproach(array=np.ma.array(
            [1,4,9,9,9, 3, 2,9,2,9,9,1,1, 3, 1,1,1,1,1],
              mask=[0,0,0,0,0, 1, 0,0,0,0,0,0,0, 1, 0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1100,1000,900,800,700,  # approach 1
                                       1000,
                                       600,550,200,100,50,20,10,  # approach 2
                                       1000,
                                       300,200,100,30,0  # approach 3
                                       ]))
        percent_stable.derive(stable, alt2app)
        self.assertEqual(len(percent_stable), 5)

        # First Approach
        self.assertEqual(percent_stable[0].name,
                         "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[0].index, 2)
        self.assertEqual(percent_stable[0].value, 75)

        # Second Approach
        self.assertEqual(percent_stable[1].name,
                         "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[1].index, 7)
        self.assertEqual(percent_stable[1].value, (3/7.0)*100)

        self.assertEqual(percent_stable[2].name,
                         "Percent Approach Stable Below 500 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[2].index, 9)
        self.assertEqual(percent_stable[2].value, 40)  # 2/5

        # Last Approach (landing)
        # test that there was an approach but non was stable
        self.assertEqual(percent_stable[3].name,
                         "Percent Approach Stable Below 1000 Ft During Last Approach")
        self.assertEqual(percent_stable[3].index, 14)
        self.assertEqual(percent_stable[3].value, 0)  # No stability == 0%

        self.assertEqual(percent_stable[4].name,
                         "Percent Approach Stable Below 500 Ft During Last Approach")
        self.assertEqual(percent_stable[4].index, 14)
        self.assertEqual(percent_stable[4].value, 0)  # No stability == 0%


class TestAltitudeAtLastAPDisengagedDuringApproach(unittest.TestCase):
    '''
    '''
    def test_can_operate(self):
        ops = AltitudeAtLastAPDisengagedDuringApproach.get_operational_combinations()
        self.assertEqual(ops, [('Altitude AAL', 'AP Disengaged Selection', 'Approach Information')])

    def test_derive_basic(self):
        alt_array = np.ma.concatenate([np.arange(10, 0, -1),
                                       np.arange(10),
                                       np.arange(10, 0, -1)])
        alt_aal = P('Altitude AAL', array=alt_array)
        ap_dis = KTI('AP Disengaged Selection',
                     items=[KeyTimeInstance(name='AP Disengaged', index=3),
                            KeyTimeInstance(name='AP Disengaged', index=7),
                            KeyTimeInstance(name='AP Disengaged', index=25)])
        apps = App('Approach Information',
                   items=[ApproachItem('TOUCH_AND_GO', slice(0, 10)),
                          ApproachItem('LANDING', slice(20, 30)),])
        node = AltitudeAtLastAPDisengagedDuringApproach()
        node.derive(alt_aal, ap_dis, apps)
        self.assertEqual(node,
                         [KeyPointValue(index=7, value=3.0, name='Altitude At Last AP Disengaged During Approach'),
                          KeyPointValue(index=25, value=5.0, name='Altitude At Last AP Disengaged During Approach')])


class TestDecelerateToStopOnRunwayDuration(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDecelerationFromTouchdownToStopOnRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = DecelerationFromTouchdownToStopOnRunway
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Groundspeed', 'Touchdown',
                                          'Landing', 'Latitude Smoothed At Touchdown', 'Longitude Smoothed At Touchdown',
            'FDR Landing Runway', 'ILS Glideslope Established',
            'ILS Localizer Established', 'Precise Positioning')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Distances: Flight


class TestGreatCircleDistance(unittest.TestCase):

    def setUp(self):
        self.node_class = GreatCircleDistance
        self.takeoff_airport = A('FDR Takeoff Airport',
                                 {"magnetic_variation":"W 01.0",
                                  "code":{"icao":"EGLL","iata":"LHR"},
                                  "elevation":83,
                                  "name":"London Heathrow",
                                  "longitude":-0.4613889999999927,
                                  "location":{"city":"London","country":"United Kingdom"},
                                  "latitude":51.47749999999999,
                                  "id":2383})
        self.landing_airport = A('FDR Landing Airport',
                                 {"magnetic_variation":"13 W",
                                  "code":{"icao":"KJFK","iata":"JFK","faa":"JFK"},
                                  "elevation":13,
                                  "name":"John F Kennedy Intl",
                                  "longitude":-73.7789,
                                  "location":{"city":"New York","country":"United States"},
                                  "latitude":40.63979999999999,
                                  "id":2794})
        self.touchdown = KTI('Touchdown', items=[KeyTimeInstance(200, 'Touchdown')])

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('FDR Takeoff Airport', 'FDR Landing Airport', 'Touchdown')))
        self.assertTrue(self.node_class.can_operate(('Latitude Smoothed At Liftoff',
                                                     'Longitude Smoothed At Liftoff',
                                                    'Latitude Smoothed At Touchdown',
                                                    'Longitude Smoothed At Touchdown',
                                                    'Touchdown')))

        self.assertTrue(self.node_class.can_operate(('FDR Takeoff Airport', 'FDR Landing Airport', 'Touchdown')))
        self.assertFalse(self.node_class.can_operate(('Latitude Smoothed At Liftoff',
                                                      'Longitude Smoothed At Liftoff',
                                                    'Latitude Smoothed At Touchdown',
                                                    'Longitude Smoothed At Touchdown')))

    def test_derive(self):
        node = self.node_class()
        node.derive(None, None, self.takeoff_airport, None, None, self.landing_airport, self.touchdown)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 200)
        self.assertAlmostEqual(node[0].value, 2991, places=0)

    def test_derive__coordinates(self):
        lat_lift = KPV('', items=[KeyPointValue(10, 51.47749999999999, '')])
        lon_lift = KPV('', items=[KeyPointValue(10, -0.4613889999999927, '')])
        lat_tdwn = KPV('', items=[KeyPointValue(200, 40.63979999999999, '')])
        lon_tdwn = KPV('', items=[KeyPointValue(200, -73.7789, '')])

        node = self.node_class()
        node.derive(lat_lift, lon_lift, None, lat_tdwn, lon_tdwn, None, self.touchdown)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 200)
        self.assertAlmostEqual(node[0].value, 2991, places=0)

    def test_derive__no_touchdonws(self):
        node = self.node_class()
        node.derive(None, None, self.takeoff_airport, None, None, self.landing_airport, None)

        self.assertEqual(len(node), 0)

    def test_derive__no_coordinates(self):
        node = self.node_class()
        node.derive(None, None, None, None, None, None, None)

        self.assertEqual(len(node), 0)


class TestDistanceFromRunwayCentrelineAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DistanceFromRunwayCentrelineAtTouchdown
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('ILS Lateral Distance', 'Touchdown')]

    def test_basic(self):
        lat = P('ILS Lateral Distance', range(10))
        tdwns = KTI(items=[KeyTimeInstance(3), KeyTimeInstance(5)])
        dist = DistanceFromRunwayCentrelineAtTouchdown()
        dist.derive(lat, tdwns)
        self.assertEqual(dist[0].value, 3.0)
        self.assertEqual(dist[0].index, 3.0)
        self.assertEqual(dist[1].value, 5.0)
        self.assertEqual(dist[1].index, 5.0)

    def test_masked(self):
        lat = P('ILS Lateral Distance', range(10))
        lat.array[5] = np.ma.masked
        tdwns = KTI(items=[KeyTimeInstance(3), KeyTimeInstance(5)])
        dist = DistanceFromRunwayCentrelineAtTouchdown()
        dist.derive(lat, tdwns)
        # Only one answer should be returned.
        self.assertEqual(len(dist), 1)

class TestDistanceFromRunwayCentrelineFromTouchdownTo60KtMax(unittest.TestCase):
    def test_can_operate(self):
        ops = DistanceFromRunwayCentrelineFromTouchdownTo60KtMax.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(ops, [('ILS Lateral Distance', 'Landing',
                                'Groundspeed', 'Touchdown')])

    def test_basic(self):
        lat = P('ILS Lateral Distance', np.ma.array([3,4,3,6,4,-7,-1,2,8,10]))
        lands=buildsection('Landing', 2, 8)
        gspd = P('Groundspeed', range(130,30,-10))
        tdwns = KTI(items=[KeyTimeInstance(3)])
        dist = DistanceFromRunwayCentrelineFromTouchdownTo60KtMax()
        dist.derive(lat, lands, gspd, tdwns)
        # The value 4 occurs at the 60kt point
        self.assertEqual(dist[0].index, 5)
        self.assertEqual(dist[0].value, -7.0)

    def test_stays_fast(self):
        lat = P('ILS Lateral Distance', np.ma.array([3,4,3,6,4,-7,-1,2,8,10]))
        lands=buildsection('Landing', 2, 8)
        gspd = P('Groundspeed', range(190,90,-10))
        tdwns = KTI(items=[KeyTimeInstance(3)])
        dist = DistanceFromRunwayCentrelineFromTouchdownTo60KtMax()
        dist.derive(lat, lands, gspd, tdwns)
        # Doesn't go below 60kts
        self.assertEqual(len(dist), 0)

    def test_abs_function(self):
        lat = P('ILS Lateral Distance', range(10))
        lat.array[3] = -20.0
        lands=buildsection('Landing', 2, 8)
        gspd = P('Groundspeed', range(130,30,-10))
        tdwns = KTI(items=[KeyTimeInstance(3)])
        dist = DistanceFromRunwayCentrelineFromTouchdownTo60KtMax()
        dist.derive(lat, lands, gspd, tdwns)
        self.assertEqual(dist[0].value, -20.0)
        self.assertEqual(dist[0].index, 3.0)


class TestDistanceFrom60KtToRunwayEnd(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromRunwayStartToTouchdown(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromTouchdownToRunwayEnd(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistancePastGlideslopeAntennaToTouchdown(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceTravelledDuringTurnback(unittest.TestCase):
    def setUp(self):
        self.node_class = DistanceTravelledDuringTurnback

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive_basic(self):
        gspd = P('Groundspeed', array=np.ma.ones(20) * 120)
        #airborne = buildsections('Airborne', [5, 16])
        toff_airport = A('FDR Takeoff Airport', value={'id':123})
        ldg_airport = A('FDR Landing Airport', value={'id':123})
        loffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        tdowns = KTI('Touchdown', items=[KeyTimeInstance(16, 'Touchdown')])
        node = self.node_class()
        node.get_derived((gspd, toff_airport, ldg_airport, loffs, tdowns))
        self.assertAlmostEqual(node[0].value, 11*120/3600.0)


class TestDistanceTravelledFollowingDiversion(unittest.TestCase):
    def setUp(self):
        self.node_class = DistanceTravelledFollowingDiversion

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive_basic(self):
        gspd = P('Groundspeed', array=np.ma.ones(400) * 120)
        destination = P('Destination', np.ma.array(['EGHH']*210 + ['EGHI']*190))
        loffs = KTI('Liftoff', items=[KeyTimeInstance(50, 'Liftoff')])
        tdowns = KTI('Touchdown', items=[KeyTimeInstance(360, 'Touchdown')])
        node = self.node_class()
        node.get_derived((gspd, destination, loffs, tdowns))
        self.assertAlmostEqual(node[0].value, 150*120/3600.0)

    def test_derive__after_touchdown(self):
        gspd = P('Groundspeed', array=np.ma.ones(40) * 120)
        destination = P('Destination', np.ma.array(['EGHH']*30 + ['EGHI']*10))
        loffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        tdowns = KTI('Touchdown', items=[KeyTimeInstance(25, 'Touchdown')])
        node = self.node_class()
        node.get_derived((gspd, destination, loffs, tdowns))
        self.assertEqual(len(node), 0)

    def test_derive__change_back(self):
        gspd = P('Groundspeed', array=np.ma.ones(400) * 120)
        destination = P('Destination', np.ma.array(['EGHI']*100 + ['EGHH']*110 + ['EGHI']*190))
        loffs = KTI('Liftoff', items=[KeyTimeInstance(50, 'Liftoff')])
        tdowns = KTI('Touchdown', items=[KeyTimeInstance(360, 'Touchdown')])
        node = self.node_class()
        node.get_derived((gspd, destination, loffs, tdowns))
        self.assertEqual(len(node), 0)


##############################################################################
# Engine Transients

class TestTakeoffRatingDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = TakeoffRatingDuration

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(),
            [('Takeoff 5 Min Rating',)]
        )

    def test_derive(self):
        toff = buildsection('Takeoff 5 Min Rating', 12, 76)
        node = self.node_class()
        node.derive(toff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 76)
        self.assertEqual(node[0].value, 64)


class TestEngGasTempOverThresholdDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngGasTempOverThresholdDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'Gas Temp': {
                'takeoff': None,
                'mcp':     800
            }
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) Gas Temp', 'Takeoff 5 Min Rating')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) Gas Temp'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(25, 875, 50), np.ones(5) * 825, np.arange(825, 600, -50)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) Gas Temp', array=array)
        eng2 = P('Eng (2) Gas Temp', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods)

        expected = KPV(
            'Eng Gas Temp Over Threshold Duration',
            items=[KeyPointValue(
                index=16.0, value=7.0,
                name='Eng Gas Temp Over MCP Duration')]
        )

        self.assertEqual(node, expected)


class TestEngN1OverThresholdDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngN1OverThresholdDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'N1': {
                'takeoff': 101,
                'mcp':     102
                },
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) N1', 'Takeoff 5 Min Rating')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) N1'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -2)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) N1', array=array)
        eng2 = P('Eng (2) N1', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods)

        expected = KPV(
            'Eng N1 Over Threshold Duration',
            items=[KeyPointValue(
                index=13.0, value=12.0,
                name='Eng N1 Over Takeoff Power Duration'),
                   KeyPointValue(
                       index=14.0, value=11.0,
                name='Eng N1 Over MCP Duration')]
        )

        self.assertEqual(node, expected)


class TestEngN2OverThresholdDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngN2OverThresholdDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'N2': {
                'takeoff': 101,
                'mcp':     102
                },
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) N2', 'Takeoff 5 Min Rating')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) N2'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -2)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) N2', array=array)
        eng2 = P('Eng (2) N2', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods)

        expected = KPV(
            'Eng N2 Over Threshold Duration',
            items=[KeyPointValue(
                index=13.0, value=12.0,
                name='Eng N2 Over Takeoff Power Duration'),
                   KeyPointValue(
                       index=14.0, value=11.0,
                name='Eng N2 Over MCP Duration')]
        )

        self.assertEqual(node, expected)


class TestEngNpOverThresholdDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngNpOverThresholdDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'Np': {
                'takeoff': 101,
                'mcp':     101
                },
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) Np', 'Takeoff 5 Min Rating')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) Np'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -2)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) Np', array=array)
        eng2 = P('Eng (2) Np', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods)

        expected = KPV(
            'Eng Np Over Threshold Duration',
            items=[KeyPointValue(
                index=13.0, value=12.0,
                name='Eng Np Over Takeoff Power Duration'),
                   KeyPointValue(
                       index=13.0, value=12.0,
                name='Eng Np Over MCP Duration')]
        )

        self.assertEqual(node, expected)


class TestEngNpMaxDuringTakeoff(unittest.TestCase):

    def setUp(self):
        self.node_class = EngNpMaxDuringTakeoff

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Takeoff 5 Min Rating', 'Eng (*) Np Max')))

    def test_derive(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 40)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98] * 3 +
                                                         [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=18, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
            KeyPointValue(index=23, value=98, name='Eng (*) Np Max During Takeoff 20 Sec'),
        ]))

    def test_derive_short_takeoff(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 21)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                             [99, 100, 99, 99, 99, 100, 99, 98, 98] * 3 +
                                                             [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=15, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
        ]))

    def test_derive_masked_data(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 40)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                             [99, 100, 99, 99, 99, 100, 99, 98] * 3 +
                                                             [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        eng_np_max.array[15:30] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=34, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
        ]))

    def test_derive_multiple_masked_slices(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 40)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98] * 3 +
                                                         [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))

        eng_np_max.array[30:35] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=18, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
            KeyPointValue(index=10, value=85, name='Eng (*) Np Max During Takeoff 20 Sec'),
        ]))

    def test_derive_all_data_masked(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 40)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                             [99, 100, 99, 99, 99, 100, 99, 98] * 3 +
                                                             [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        eng_np_max.array[10:40] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive_not_enough_high_samples(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 20)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83,
                                                          81, 82, 83, 85, 90, 91, 92,
                                                          99, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=15, value=86, name='Eng (*) Np Max During Takeoff 5 Sec'),
        ]))

    def test_derive_hz_2(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 55)
        eng_np_max=P('Eng (*) Np Max',
                     frequency=2,
                     array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                       [85, 90, 92, 95, 96, 99, 98, 98] +
                                       [99, 100, 99, 99, 99, 100, 99, 98] * 4 +
                                       [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=25, value=98, name='Eng (*) Np Max During Takeoff 5 Sec'),
                KeyPointValue(index=12, value=92, name='Eng (*) Np Max During Takeoff 20 Sec'),
        ]))

    def test_derive_same_values_returned(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 40)
        eng_np_max=P('Eng (*) Np Max', array=np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                         [85, 90, 92, 95, 96, 99] +
                                                             [99, 100, 99, 99, 99, 100, 99, 99] * 3 +
                                                             [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=18, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
                KeyPointValue(index=15, value=99, name='Eng (*) Np Max During Takeoff 20 Sec'),
        ]))

    def test_derive_with_go_around(self):
        go_arounds=buildsection('Go Around 5 Min Rating', 10, 40)
        takeoffs=buildsection('Takeoff 5 Min Rating', 60, 90)
        eng_np_max=P('Eng (*) Np Max', array=np.tile(np.ma.array([80, 81, 82, 83, 83, 82, 83, 81, 82, 83] +
                                                                 [85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98] * 3 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]), 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_np_max, takeoffs, go_arounds)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=68, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
            KeyPointValue(index=73, value=98, name='Eng (*) Np Max During Takeoff 20 Sec'),
            KeyPointValue(index=18, value=99, name='Eng (*) Np Max During Takeoff 5 Sec'),
            KeyPointValue(index=23, value=98, name='Eng (*) Np Max During Takeoff 20 Sec'),
        ]))


class TestEngTorqueOverThresholdDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngTorqueOverThresholdDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'Torque': {
                'takeoff': 95,
                'mcp':     90
                },
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate_heli(self, lookup_table):
        nodes = ('Eng (1) Torque', 'Takeoff 5 Min Rating', 'All Engines Operative')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
            'ac_type': helicopter
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) Torque'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) Torque', 'Takeoff 5 Min Rating')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) Torque'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive_1(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -2)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) Torque', array=array)
        eng2 = P('Eng (2) Torque', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods, None)

        expected = KPV(
            'Eng Torque Over Threshold Duration',
            items=[KeyPointValue(
                index=10.0, value=15.0,
                name='Eng Torque Over Takeoff Power Duration'),
                   KeyPointValue(
                       index=8.0, value=17.0,
                name='Eng Torque Over MCP Duration')]
        )

        self.assertEqual(node, expected)

    @patch('analysis_engine.key_point_values.at')
    def test_derive_2(self, lookup_table):

        lookup_table.get_engine_map.return_value = {'Torque': {'mcp': 90, 'takeoff': 92}}

        eng1 = P('Eng (1) Torque', array=load_compressed(
            os.path.join(test_data_path, 'EngTorqueOverThresholdDuration_eng_1_torque.npz')))
        eng2 = P('Eng (2) Torque', array=load_compressed(
            os.path.join(test_data_path, 'EngTorqueOverThresholdDuration_eng_2_torque.npz')))
        takeoff = buildsection('Takeoff 5 Min Rating', 526.8325733573796, 607.84375)
        mcp = buildsection('Maximum Continuous Power', 607.8325733573796, 5360.8325733573793)

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods, None)

        expected = [KeyPointValue(*a) for a in [
            (538, 44.0, 'Eng Torque Over Takeoff Power Duration'),
            (609, 3.0, 'Eng Torque Over MCP Duration'),
            (631, 184.0, 'Eng Torque Over MCP Duration'),
            (820, 72.0, 'Eng Torque Over MCP Duration'),
            (962, 4.0, 'Eng Torque Over MCP Duration'),
            (1021, 11.0, 'Eng Torque Over MCP Duration'),
            (1037, 10.0, 'Eng Torque Over MCP Duration'),
            (1149, 15.0, 'Eng Torque Over MCP Duration'),
            (1642, 190.0, 'Eng Torque Over MCP Duration'),
            (1836, 441.0, 'Eng Torque Over MCP Duration'),
            (2286, 222.0, 'Eng Torque Over MCP Duration'),
            (2565, 1813.0, 'Eng Torque Over MCP Duration')]]

        self.assertEqual(list(node), expected)

    @patch('analysis_engine.key_point_values.at')
    def test_derive_heli(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -2)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) Torque', array=array)
        eng2 = P('Eng (2) Torque', array=array)
        takeoff = buildsection('Takeoff 5 Min Rating', 5, 25)
        mcp = buildsection('Maximum Continuous Power', 5, 25)
        all_eng = M('All Engines Operative', np.ma.concatenate((np.zeros(15), np.ones(38))), values_mapping={0:'-', 1:'AEO'})

        node = self.node_class()
        node.derive(eng1, eng2, None, None, takeoff, mcp, None, self.engine_series, self.engine_type, self.mods, all_eng)

        expected = KPV(
            'Eng Torque Over Threshold Duration',
            items=[KeyPointValue(
                index=15.0, value=10.0,
                name='Eng Torque Over Takeoff Power Duration'),
                   KeyPointValue(
                       index=15.0, value=10.0,
                name='Eng Torque Over MCP Duration')]
        )

        self.assertEqual(node, expected)


class TestEngTorqueLimitExceedanceWithOneEngineInoperativeDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngTorqueLimitExceedanceWithOneEngineInoperativeDuration
        self.mods = A('Modifications', [])
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_type = A('Engine Type', 'PW124B')
        self.engine_series = A('Engine Series', 'PW100')
        self.engine_thresholds = {
            'Torque': {
                'continuous': (None, 85),
                'low':        ( 120, 85),
                'hi':         (   8, 95),
                'transient':  (   3, 100)
                },
        }

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, lookup_table):
        nodes = ('Eng (1) Torque', 'One Engine Inoperative')
        kwargs = {
            'eng_series': self.engine_series,
            'eng_type': self.engine_type,
            'mods': self.mods,
            'ac_type': helicopter
        }
        lookup_table.get_engine_map.return_value = self.engine_thresholds
        self.assertTrue(self.node_class.can_operate(nodes, **kwargs))
        self.assertFalse(self.node_class.can_operate(('Eng (1) Torque'), **kwargs))
        # No lookup table found
        lookup_table.get_engine_map.side_effect = KeyError("No engine threshods for 'PW100', series 'PW124B', type '[]' mods.")
        self.assertFalse(self.node_class.can_operate(nodes, **kwargs))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, lookup_table):

        lookup_table.get_engine_map.return_value = self.engine_thresholds

        array = np.ma.concatenate((np.arange(75, 115, 2), np.ones(5) * 115, np.arange(115, 60, -1)))
        # use two arrays the same to ensure only one KPV created.
        eng1 = P('Eng (1) Torque', array=array)
        eng2 = P('Eng (2) Torque', array=array)
        all_eng = M('One Engine Inoperable', np.ma.concatenate((np.zeros(31), np.ones(49))), values_mapping={0:'-', 1:'OEI'})

        node = self.node_class()
        node.derive(eng1, eng2, None, None, all_eng, self.engine_series, self.engine_type, self.mods)

        node_name = 'Eng Torque Limit Exceedance With One Engine Inoperative Duration'
        expected = KPV(
            node_name,
            items=[KeyPointValue(
                index=46.0, value=9.0,
                name=node_name),
                   KeyPointValue(
                       index=41.0, value=4.0,
                name=node_name)]
        )

        self.assertEqual(node, expected)


class TestEngTorqueMaxDuringMaximumContinuousPower(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTorqueMaxDuringMaximumContinuousPower

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Eng (*) Torque Max', 'Maximum Continuous Power')))

    def test_derive(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                  95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                                 [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 20 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 5 Min'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Min'),
        ]))

    def test_derive_short_mcp(self):
        mcp=buildsection('Maximum Continuous Power', 20, 400)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                  95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                                 [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 20 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 5 Min'),
        ]))

    def test_derive_masked_data(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                  95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                                 [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        eng_torque_max.array[60:] =np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 20 Sec'),
        ]))

    def test_derive_multiple_masked_slices(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                  95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                                 [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))

        eng_torque_max.array[350:400] = np.ma.masked
        eng_torque_max.array[450:500] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 20 Sec'),
                KeyPointValue(index=23, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 5 Min'),
        ]))

    def test_derive_all_data_masked(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array(([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                   95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                                  [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                  [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]), mask=True))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive_hz_2(self):
        mcp=buildsection('Maximum Continuous Power', 40, 1400)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                                  95, 92, 93, 89, 87, 85, 82, 83, 80, 79] * 2  +
                                                                 [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 140 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0] * 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=43, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Sec'),
                KeyPointValue(index=43, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 20 Sec'),
                KeyPointValue(index=43, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 5 Min'),
                KeyPointValue(index=43, value=75, name='Eng (*) Torque Max During Maximum Continuous Power 10 Min'),
        ]))


class TestEngTorqueMaxDuringTakeoff(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTorqueMaxDuringTakeoff

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Eng (*) Torque Max', 'Takeoff 5 Min Rating')))

    def test_derive(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 5 Min'),
        ]))

    def test_derive_short_takeoff(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 25)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
        ]))

    def test_derive_masked_data(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        eng_torque_max.array[50:360] = np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
                KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
        ]))

    def test_derive_multiple_masked_slices(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))

        eng_torque_max.array[200:250] = np.ma.masked
        eng_torque_max.array[100:150] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
        ]))

    def test_derive_all_data_masked(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array(([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                  [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]), mask=True))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive_not_enough_high_samples(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 400)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 98] * 2 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82] * 40))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=24, value=86, name='Eng (*) Torque Max During Takeoff 10 Sec'),
                KeyPointValue(index=26, value=80, name='Eng (*) Torque Max During Takeoff 20 Sec'),
                KeyPointValue(index=26, value=80, name='Eng (*) Torque Max During Takeoff 5 Min'),
        ]))

    def test_derive_hz_2(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 20, 720)
        eng_torque_max=P('Eng (*) Torque Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] * 2 +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 70 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82] * 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=27, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
                KeyPointValue(index=27, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
                KeyPointValue(index=27, value=98, name='Eng (*) Torque Max During Takeoff 5 Min'),
        ]))

    def test_derive_with_go_around(self):
        go_arounds=buildsection('Go Around 5 Min Rating', 10, 360)
        takeoffs=buildsection('Takeoff 5 Min Rating', 380, 720)
        eng_torque_max=P('Eng (*) Torque Max', array=np.tile(np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                         [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]), 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_torque_max, takeoffs, go_arounds)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=387, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
            KeyPointValue(index=387, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
            KeyPointValue(index=387, value=98, name='Eng (*) Torque Max During Takeoff 5 Min'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 10 Sec'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 20 Sec'),
            KeyPointValue(index=17, value=98, name='Eng (*) Torque Max During Takeoff 5 Min'),
        ]))


class TestEngN2DuringTakeoffForXSecMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN2DuringTakeoffForXSecMax

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Eng (*) N2 Max', 'Takeoff 5 Min Rating')))

    def test_derive(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 5 Min Max'),
        ]))

    def test_derive_short_takeoff(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 25)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
        ]))

    def test_derive_masked_data(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))
        eng_N2_max.array[50:360] = np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
                KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
        ]))

    def test_derive_multiple_masked_slices(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]))

        eng_N2_max.array[200:250] = np.ma.masked
        eng_N2_max.array[100:150] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
        ]))

    def test_derive_all_data_masked(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 360)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array(([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                          [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]), mask=True))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive_not_enough_high_samples(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 10, 400)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 98] * 2 +
                                                         [95, 94, 92, 89, 86, 82, 80, 80, 81, 82] * 40))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=24, value=86, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
                KeyPointValue(index=26, value=80, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
                KeyPointValue(index=26, value=80, name='Eng (*) N2 During Takeoff For 5 Min Max'),
        ]))

    def test_derive_hz_2(self):
        takeoffs=buildsection('Takeoff 5 Min Rating', 20, 720)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] * 2 +
                                                         [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 70 +
                                                                     [95, 94, 92, 89, 86, 82, 80, 80, 81, 82] * 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=27, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
                KeyPointValue(index=27, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
                KeyPointValue(index=27, value=98, name='Eng (*) N2 During Takeoff For 5 Min Max'),
        ]))

    def test_derive_with_go_around(self):
        go_arounds=buildsection('Go Around 5 Min Rating', 10, 360)
        takeoffs=buildsection('Takeoff 5 Min Rating', 380, 720)
        eng_N2_max=P('Eng (*) N2 Max', array=np.tile(np.ma.array([80, 81, 82, 83, 85, 90, 92, 95, 96, 99] +
                                                                 [99, 100, 99, 99, 99, 100, 99, 98, 99, 98] * 35 +
                                                                 [95, 94, 92, 89, 86, 82, 80, 80, 81, 82]), 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, takeoffs, go_arounds)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=387, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
            KeyPointValue(index=387, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
            KeyPointValue(index=387, value=98, name='Eng (*) N2 During Takeoff For 5 Min Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 10 Sec Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 20 Sec Max'),
            KeyPointValue(index=17, value=98, name='Eng (*) N2 During Takeoff For 5 Min Max'),
        ]))


class TestEngN2DuringMaximumContinuousPowerForXSecMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN2DuringMaximumContinuousPowerForXSecMax

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Eng (*) N2 Max', 'Maximum Continuous Power')))

    def test_derive(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                          95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                         [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 20 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 5 Min Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Min Max'),
        ]))

    def test_derive_short_mcp(self):
        mcp=buildsection('Maximum Continuous Power', 20, 400)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                          95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                         [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 20 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 5 Min Max'),
        ]))

    def test_derive_masked_data(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                          95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                         [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))
        eng_N2_max.array[60:] =np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 20 Sec Max'),
        ]))

    def test_derive_multiple_masked_slices(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                          95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                         [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]))

        eng_N2_max.array[350:400] = np.ma.masked
        eng_N2_max.array[450:500] = np.ma.masked

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 20 Sec Max'),
                KeyPointValue(index=23, value=75, name='Eng (*) N2 During Maximum Continuous Power For 5 Min Max'),
        ]))

    def test_derive_all_data_masked(self):
        mcp=buildsection('Maximum Continuous Power', 20, 700)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array(([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                           95, 92, 93, 89, 87, 85, 82, 83, 80, 79] +
                                                          [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 70 +
                                                                  [75, 72, 68, 63, 58, 40, 30, 20, 10, 0]), mask=True))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive_hz_2(self):
        mcp=buildsection('Maximum Continuous Power', 40, 1400)
        eng_N2_max=P('Eng (*) N2 Max', array=np.ma.array([80, 81, 85, 90, 95, 96, 99, 100, 100, 99,
                                                          95, 92, 93, 89, 87, 85, 82, 83, 80, 79] * 2  +
                                                         [80, 81, 79, 75, 78, 81, 82, 79, 80, 79] * 140 +
                                                                 [75, 72, 68, 63, 58, 40, 30, 20, 10, 0] * 2))
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(eng_N2_max, mcp)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=43, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Sec Max'),
                KeyPointValue(index=43, value=75, name='Eng (*) N2 During Maximum Continuous Power For 20 Sec Max'),
                KeyPointValue(index=43, value=75, name='Eng (*) N2 During Maximum Continuous Power For 5 Min Max'),
                KeyPointValue(index=43, value=75, name='Eng (*) N2 During Maximum Continuous Power For 10 Min Max'),
        ]))


##############################################################################
# Engine Bleed


class TestEngBleedValvesAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngBleedValvesAtLiftoff
        self.operational_combinations = [('Liftoff', 'Eng Bleed Open'),]

    def test_derive(self):
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(name='Liftoff', index=3)])
        values_mapping = {0: 'Closed', 1: 'Open'}
        bleed = M('Eng Bleed Open', array=np.ma.masked_array([0, 1, 1, 1, 0]), values_mapping=values_mapping)
        node = EngBleedValvesAtLiftoff()
        node.derive(liftoff, bleed)
        self.assertEqual(node, KPV('Eng Bleed Valves At Liftoff', items=[
            KeyPointValue(name='Eng Bleed Valves At Liftoff', index=3, value=1),
        ]))


##############################################################################
# Engine EPR


class TestEngEPRDuringApproachMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRDuringApproachMax

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('Eng (*) EPR Max', 'Approach')]
        self.assertEqual(ops, expected)

    def test_derive(self):
        approaches = buildsection('Approach', 70, 120)
        epr_array = np.round(10 + np.ma.array(10 * np.sin(np.arange(0, 12.6, 0.1))))
        epr = P(name='Eng (*) EPR Max', array=epr_array)
        node = self.node_class()
        node.derive(epr, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 76)
        self.assertEqual(node[0].value, 20)
        self.assertEqual(node[0].name, 'Eng EPR During Approach Max')


class TestEngEPRDuringApproachMin(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRDuringApproachMin

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('Eng (*) EPR Min', 'Approach')]
        self.assertEqual(ops, expected)


    def test_derive(self):
        approaches = buildsection('Approach', 70, 120)
        epr_array = np.round(10 + np.ma.array(10 * np.sin(np.arange(0, 12.6, 0.1))))
        epr = P(name='Eng (*) EPR Max', array=epr_array)
        node = self.node_class()
        node.derive(epr, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 107)
        self.assertEqual(node[0].value, 0)
        self.assertEqual(node[0].name, 'Eng EPR During Approach Min')


class TestEngEPRLowDuringTakeoff(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRLowDuringTakeoff
        self.operational_combinations = [('Eng (*) EPR Min', 'Takeoff')]
        self.function = max_value

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('Eng (*) EPR Min', 'Takeoff')]
        self.assertEqual(ops, expected)

    def test_derive(self):
        takeoff = buildsection('Takeoff', 15, 50)
        epr_array = np.ma.array([1.00, 1.00, 1.00, 1.00, 1.00,
                                 1.01, 1.01, 1.00, 1.00, 1.00,
                                 1.00, 1.00, 1.00, 1.00, 1.00,
                                 1.01, 1.01, 1.01, 1.01, 1.02,
                                 1.06, 1.12, 1.23, 1.38, 1.41,
                                 1.42, 1.43, 1.42, 1.42, 1.42,
                                 1.42, 1.42, 1.42, 1.42, 1.42,
                                 1.42, 1.42, 1.42, 1.41, 1.41,
                                 1.41, 1.41, 1.41, 1.41, 1.41,
                                 1.40, 1.40, 1.40, 1.40, 1.40,
                                 1.39, 1.39, 1.39, 1.38, 1.38,
                                 1.38, 1.38, 1.38, 1.38, 1.38,
                                 ])
        epr = P(name='Eng (*) EPR Min', array=epr_array)
        node = self.node_class()
        node.derive(epr, takeoff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 26)
        self.assertEqual(node[0].value, 1.43)


class TestEngEPRDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTaxiMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRDuringTaxiOutMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTaxiOutMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Taxi Out')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRDuringTaxiInMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTaxiInMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Taxi In')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRFor5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngEPRFor5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRFor5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngEPRFor5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngEPRFor5SecDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngEPRFor5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPR500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPR500To50FtMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPR500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPR500To50FtMin
        self.operational_combinations = [('Eng (*) EPR Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRFor5Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRFor5Sec1000To500FtMin
        self.operational_combinations = [('Eng (*) EPR Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRFor5Sec500To50FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRFor5Sec500To50FtMin
        self.operational_combinations = [('Eng (*) EPR Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRAtTOGADuringTakeoffMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRAtTOGADuringTakeoffMax
        self.operational_combinations = [('Takeoff And Go Around', 'Eng (*) EPR Max', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRAtTOGADuringTakeoffMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngTPRAtTOGADuringTakeoffMin
        self.operational_combinations = [('Takeoff And Go Around', 'Eng (*) TPR Min', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTPRDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRFor5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTPRFor5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTPRDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRFor5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTPRFor5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTPRDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) TPR Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngTPRFor5SecDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTPRFor5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) TPR Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRExceedEPRRedlineDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRExceedEPRRedlineDuration
        array = np.ma.concatenate((np.zeros(5), np.arange(10), np.ones(10) * 10))
        self.array = np.ma.concatenate((array, array[::-1]))

    def test_derive(self):
        epr_1 = P(name='Eng (1) EPR', array=self.array.copy())
        epr_1_red = P(name='Eng (1) EPR Redline', array=self.array.copy()+20)
        epr_2 = P(name='Eng (2) EPR', array=self.array.copy())
        epr_2_red = P(name='Eng (2) EPR Redline', array=self.array.copy()+20)
        epr_2.array[13:17] = 35

        node = self.node_class()
        node.derive(epr_1, epr_1_red, epr_2, epr_2_red, *[None]*4)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].name, 'Eng EPR Exceeded EPR Redline Duration')


##############################################################################
# Engine Fire


class TestEngFireWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Eng (*) Fire'
        self.phase_name = 'Airborne'
        self.node_class = EngFireWarningDuration
        self.values_mapping = {0: '-', 1: 'Fire'}

        self.basic_setup()


##############################################################################
# APU On


class TestAPUOnDuringFlightDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = APUOnDuringFlightDuration

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [
            ('APU On', 'Airborne')])

    def test_derive__start_before_liftoff(self):
        apu = MultistateDerivedParameterNode(
            name='APU On',
            array=np.ma.array([0] * 5 + [1] * 10 + [0] * 15),
            values_mapping={0: '-', 1: 'On'}
        )
        airborne = S(items=[Section('Airborne',
                                    slice(13, 20), 10, 20)])
        node = self.node_class()
        node.derive(apu, airborne)
        # The APU On started 3 seconds before the liftoff and lasted 5 seconds,
        # which means there were 2 seconds of APU On in air
        self.assertEqual(node[0].value, 2)
        # The index of the LAST sample where apu.array == 'On'
        self.assertEqual(node[0].index, 15)

    def test_derive__start_in_flight(self):
        apu = MultistateDerivedParameterNode(
            name='APU On',
            array=np.ma.array([0] * 15 + [1] * 10 + [0] * 5),
            values_mapping={0: '-', 1: 'On'}
        )
        airborne = S(items=[Section('Airborne',
                                    slice(13, 20), 10, 20)])
        node = self.node_class()
        node.derive(apu, airborne)
        # The APU On started mid flight and lasted 5 seconds
        self.assertEqual(node[0].value, 5)
        # The index of the FIRST sample where apu.array == 'On'
        self.assertEqual(node[0].index, 15)

##############################################################################
# APU Fire


class TestAPUFireWarningDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = APUFireWarningDuration

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        simple = ('APU Fire',)
        bottles = ('Fire APU Single Bottle System', 'Fire APU Dual Bottle System')
        self.assertTrue(simple in opts)
        self.assertTrue(bottles in opts)

    def test_derive_basic(self):
        values_mapping = {
            0: '-',
            1: 'Fire',
        }
        single_fire = M(name='Fire APU Single Bottle System',
                        array=np.ma.zeros(10),
                        values_mapping=values_mapping)
        single_fire.array[5:7] = 'Fire'

        node = self.node_class()
        node.derive(None, single_fire, None)

        self.assertEqual(node[0].name, 'APU Fire Warning Duration')
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(len(node), 1)

        # Test simple case
        node = self.node_class()
        node.derive(single_fire, None, None)

        self.assertEqual(node[0].name, 'APU Fire Warning Duration')
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(len(node), 1)


##############################################################################
# Engine Shutdown


class TestEngShutdownDuringFlightDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngShutdownDuringFlightDuration
        self.operational_combinations = [('Eng (*) All Running', 'Airborne')]

    def test_derive(self):
        eng_running = M(
            array=np.ma.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            values_mapping={0: 'Not Running', 1: 'Running'},
        )
        airborne = S(items=[Section('Airborne', slice(4, 40), 4.1, 30.1)])
        node = self.node_class(frequency=2)
        node.derive(eng_running=eng_running, airborne=airborne)
        # Note: Should only be single KPV (as must be greater than 4 seconds)
        self.assertEqual(node, [
            KeyPointValue(index=20, value=10.0, name='Eng Shutdown During Flight Duration'),
        ])


##############################################################################


class TestSingleEngineDuringTaxiInDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = SingleEngineDuringTaxiInDuration

        self.operational_combinations = [
            ('Eng (*) All Running', 'Eng (*) Any Running', 'Taxi In')]

    def test_derive(self):
        any_eng_array = np.ma.array([0, 0, 1, 1, 1, 1, 1])
        any_eng = P('Eng (*) Any Running', array=any_eng_array)
        all_eng_array = np.ma.array([0, 0, 0, 0, 0, 1, 1])
        all_eng = P('Eng (*) All Running', array=all_eng_array)
        taxi_in = S(items=[Section('Taxi In', slice(1, 6), 1, 6)])
        node = self.node_class()
        node.derive(all_eng, any_eng, taxi_in)
        expected = KPV(
            'Single Engine During Taxi In Duration',
            items=[KeyPointValue(
                index=2.0, value=3.0,
                name='Single Engine During Taxi In Duration')]
        )
        self.assertEqual(node, expected)


class TestSingleEngineDuringTaxiOutDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = SingleEngineDuringTaxiOutDuration

        self.operational_combinations = [
            ('Eng (*) All Running', 'Eng (*) Any Running', 'Taxi Out')]

    def test_derive(self):
        any_eng_array = np.ma.array([0, 0, 1, 1, 1, 1, 1])
        any_eng = P('Eng (*) Any Running', array=any_eng_array)
        all_eng_array = np.ma.array([0, 0, 0, 0, 0, 1, 1])
        all_eng = P('Eng (*) All Running', array=all_eng_array)
        taxi_out = S(items=[Section('Taxi Out', slice(1, 6), 1, 6)])
        node = self.node_class()
        node.derive(all_eng, any_eng, taxi_out)
        expected = KPV(
            'Single Engine During Taxi Out Duration',
            items=[KeyPointValue(
                index=2.0, value=3.0,
                name='Single Engine During Taxi Out Duration')]
        )
        self.assertEqual(node, expected)


##############################################################################
# Engine Gas Temperature


class TestEngGasTempDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngGasTempDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngGasTempFor5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngGasTempFor5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngGasTempDuringGoAround5MinRatingMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngGasTempDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempFor5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngGasTempFor5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngGasTempDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngGasTempFor5SecDuringMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngGasTempFor5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringMaximumContinuousPowerForXMinMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringMaximumContinuousPowerForXMinMax
        self.operational_combinations =  [
            ('Takeoff 5 Min Rating', 'Eng (*) Gas Temp Max', 'Airborne'),
            ('Takeoff 5 Min Rating', 'Eng (*) Gas Temp Max',
             'Go Around 5 Min Rating', 'Airborne')
        ]
        self.eng_egt_max = load(
            os.path.join(test_data_path, 'ebe456663820_eng_egt_max.nod'))
        self.to_ratings = load(
            os.path.join(test_data_path, 'ebe456663820_takeoffs.nod'))
        self.ga_ratings = load(
            os.path.join(test_data_path, 'ebe456663820_go_arounds.nod'))
        self.airborne = load(
            os.path.join(test_data_path, 'ebe456663820_airborne.nod'))


    def test_derive(self):
        node = self.node_class()
        node.derive(self.eng_egt_max, self.to_ratings,
                    self.ga_ratings, self.airborne)

        expected_results = [
            (1438.0, 724.0, 'Eng Gas Temp During Maximum Continuous Power For 3 Min Max'),
             (3362.0, 724.0, 'Eng Gas Temp During Maximum Continuous Power For 3 Min Max'),
             (1208.0, 709.0, 'Eng Gas Temp During Maximum Continuous Power For 5 Min Max'),
             (3277.0, 716.0, 'Eng Gas Temp During Maximum Continuous Power For 5 Min Max'),
             (1516.0, 735.0, 'Eng Gas Temp During Maximum Continuous Power For 10 Sec Max'),
             (3456.0, 738.0, 'Eng Gas Temp During Maximum Continuous Power For 10 Sec Max'),
             (1510.0, 734.0, 'Eng Gas Temp During Maximum Continuous Power For 20 Sec Max'),
             (3446.0, 737.0, 'Eng Gas Temp During Maximum Continuous Power For 20 Sec Max'),
        ]

        self.assertEqual(len(node), 8)
        self.assertEqual(node.name, 'Eng Gas Temp During Maximum Continuous Power For X Min Max')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)


    def test_derive_no_goaround(self):
        node = self.node_class()
        node.derive(self.eng_egt_max, self.to_ratings,
                    S('Go Around 5 Min Rating'), self.airborne)

        expected_results = [
            (1438.0, 724.0, 'Eng Gas Temp During Maximum Continuous Power For 3 Min Max'),
             (3277.0, 716.0, 'Eng Gas Temp During Maximum Continuous Power For 5 Min Max'),
             (2927.0, 804.0, 'Eng Gas Temp During Maximum Continuous Power For 10 Sec Max'),
             (2918.0, 777.0, 'Eng Gas Temp During Maximum Continuous Power For 20 Sec Max'),
        ]
        self.assertEqual(len(node), 4)
        self.assertEqual(node.name, 'Eng Gas Temp During Maximum Continuous Power For X Min Max')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)

    def test_derive_no_goaround_helicopter(self):
        '''
        Helicopters do not generate go-around phases, so the argument to
        derive would be None instead of an empty list
        '''
        node = self.node_class()
        node.derive(self.eng_egt_max, self.to_ratings,
                    None, self.airborne)

        expected_results = [
            (1438.0, 724.0, 'Eng Gas Temp During Maximum Continuous Power For 3 Min Max'),
             (3277.0, 716.0, 'Eng Gas Temp During Maximum Continuous Power For 5 Min Max'),
             (2927.0, 804.0, 'Eng Gas Temp During Maximum Continuous Power For 10 Sec Max'),
             (2918.0, 777.0, 'Eng Gas Temp During Maximum Continuous Power For 20 Sec Max'),
        ]

        self.assertEqual(len(node), 4)
        self.assertEqual(node.name, 'Eng Gas Temp During Maximum Continuous Power For X Min Max')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)

class TestEngGasTempMaxDuringTakeoffMaxMaintained(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = EngGasTempMaxDuringTakeoffMaxMaintained
        self.operational_combinations = [
            ('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating',),
            ('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating' )
        ]
        self.eng_egt_max = load(
            os.path.join(test_data_path, 'ebe456663820_eng_egt_max.nod'))
        self.takeoffs = load(
            os.path.join(test_data_path, 'ebe456663820_takeoffs.nod'))
        self.go_arounds = load(
            os.path.join(test_data_path, 'ebe456663820_go_arounds.nod'))


    def test_derive(self):
        node = self.node_class()
        node.derive(self.eng_egt_max, self.takeoffs, self.go_arounds)

        expected_results = [
            (926.8359375, 722.3524525, 'Eng Gas Temp Max During Takeoff 5 Sec Max Maintained'),
            (2934.5546875, 810.2792775, 'Eng Gas Temp Max During Takeoff 5 Sec Max Maintained'),
            (923.8359375, 721.096355, 'Eng Gas Temp Max During Takeoff 10 Sec Max Maintained'),
            (2929.5546875, 805.568911875, 'Eng Gas Temp Max During Takeoff 10 Sec Max Maintained'),
            (913.8359375, 718.898184375, 'Eng Gas Temp Max During Takeoff 20 Sec Max Maintained'),
            (2919.5546875, 784.52927875, 'Eng Gas Temp Max During Takeoff 20 Sec Max Maintained'),
            (2740.5546875, 444.318, 'Eng Gas Temp Max During Takeoff 5 Min Max Maintained'),
        ]

        self.assertEqual(len(node), 7)
        self.assertEqual(node.name, 'Eng Gas Temp Max During Takeoff Max Maintained')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)

    def test_derive_no_goaround(self):
        node = self.node_class()
        node.derive(self.eng_egt_max, self.takeoffs, S('Go Around 5 Min Rating'))

        expected_results = [
            (926.8359375, 722.3524525, 'Eng Gas Temp Max During Takeoff 5 Sec Max Maintained'),
            (923.8359375, 721.096355, 'Eng Gas Temp Max During Takeoff 10 Sec Max Maintained'),
            (913.8359375, 718.898184375, 'Eng Gas Temp Max During Takeoff 20 Sec Max Maintained'),
        ]

        self.assertEqual(len(node), 3)
        self.assertEqual(node.name, 'Eng Gas Temp Max During Takeoff Max Maintained')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)

    def test_derive_heli(self):
        '''
        Helicopters do not generate go-around phases, so the argument to
        derive would be None instead of an empty list
        '''
        node = self.node_class()
        node.derive(self.eng_egt_max, self.takeoffs, None)

        expected_results = [
            (926.8359375, 722.3524525, 'Eng Gas Temp Max During Takeoff 5 Sec Max Maintained'),
            (923.8359375, 721.096355, 'Eng Gas Temp Max During Takeoff 10 Sec Max Maintained'),
            (913.8359375, 718.898184375, 'Eng Gas Temp Max During Takeoff 20 Sec Max Maintained'),
        ]

        self.assertEqual(len(node), 3)
        self.assertEqual(node.name, 'Eng Gas Temp Max During Takeoff Max Maintained')
        for kpv_node, expected  in zip(node, expected_results):
            expected_index, expected_value, expected_name = expected
            self.assertAlmostEqual(kpv_node.index, expected_index, places=0)
            self.assertAlmostEqual(kpv_node.value, expected_value, places=0)
            self.assertAlmostEqual(kpv_node.name, expected_name, places=0)


class TestEngGasTempDuringEngStartMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringEngStartMax
        self.operational_combinations = [('Eng (1) Gas Temp', 'Eng (1) N2', 'Eng (1) N3')]

    def test_derive(self):
        eng_starts = EngStart('Eng Start', items=[
            KeyTimeInstance(163, 'Eng (1) Start'),
            KeyTimeInstance(98, 'Eng (2) Start'),
        ])
        eng_1_egt = load(os.path.join(test_data_path, 'eng_start_eng_1_egt.nod'))
        eng_2_egt = load(os.path.join(test_data_path, 'eng_start_eng_2_egt.nod'))
        eng_1_n3 = load(os.path.join(test_data_path, 'eng_start_eng_1_n3.nod'))
        eng_2_n3 = load(os.path.join(test_data_path, 'eng_start_eng_2_n3.nod'))
        node = EngGasTempDuringEngStartMax()
        node.derive(eng_1_egt, eng_2_egt, None, None, eng_1_n3, eng_2_n3, None,
                    None, None, None, None, None, None, None, None, None, eng_starts)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 174)
        self.assertEqual(node[0].value, 303)
        self.assertEqual(node[1].index, 99)
        self.assertEqual(node[1].value, 333)


class TestEngGasTempDuringEngStartForXSecMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringEngStartForXSecMax
        self.operational_combinations = [('Eng (*) Gas Temp Max',
                                          'Eng (*) N2 Min',
                                          'Takeoff Turn Onto Runway')]

    def test_derive(self):
        eng_egt_max = load(
            os.path.join(test_data_path, 'ebe456663820_eng_egt_max.nod'))
        toff_turn_rwy = load(
            os.path.join(test_data_path, 'ebe456663820_toff_turn_rwy.nod'))
        eng_n2_min = load(
            os.path.join(test_data_path, 'ebe456663820_eng_n2_min.nod'))

        node = self.node_class()
        node.derive(eng_egt_max, eng_n2_min, toff_turn_rwy)

        self.assertEqual(
            node,
            KPV('Eng Gas Temp During Eng Start For X Sec Max',
                items=[
                    KeyPointValue(index=224, value=513,
                                  name='Eng Gas Temp During Eng Start For 5 Sec Max'),
                    KeyPointValue(index=1, value=510,
                                  name='Eng Gas Temp During Eng Start For 10 Sec Max'),
                    KeyPointValue(index=1, value=510,
                                  name='Eng Gas Temp During Eng Start For 20 Sec Max'),
                    KeyPointValue(index=1, value=510,
                                  name='Eng Gas Temp During Eng Start For 40 Sec Max'),
                ])
        )


class TestEngGasTempDuringFlightMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngGasTempDuringFlightMin
        self.operational_combinations = [('Eng (*) Gas Temp Min', 'Airborne')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempExceededEngGasTempRedlineDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngGasTempExceededEngGasTempRedlineDuration
        array = np.ma.concatenate((np.zeros(5), np.arange(10), np.ones(10) * 10))
        self.array = np.ma.concatenate((array, array[::-1]))

    def test_derive(self):
        egt_1 = P(name='Eng (1) Gas Temp', array=self.array.copy())
        egt_1_red = P(name='Eng (1) Gas Temp Redline', array=self.array.copy()+20)
        egt_2 = P(name='Eng (2) Gas Temp', array=self.array.copy())
        egt_2_red = P(name='Eng (2) Gas Temp Redline', array=self.array.copy()+20)
        egt_2.array[13:17] = 35

        node = self.node_class()
        node.derive(egt_1, egt_1_red, egt_2, egt_2_red, *[None]*4)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].name, 'Eng Gas Temp Exceeded Eng Gas Temp Redline Duration')


class TestEngGasTempAboveNormalMaxLimitDuringTakeoff5MinRatingDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngGasTempAboveNormalMaxLimitDuringTakeoff5MinRatingDuration

    def test_can_operate(self):
        nodes = ('Eng (1) Gas Temp', 'Takeoff 5 Min Rating')
        engine = A('Engine Series', value='CFM56-5A')
        self.assertFalse(self.node_class.can_operate(nodes, eng_series=engine))
        engine = A('Engine Series', value='CFM56-3')
        self.assertTrue(self.node_class.can_operate(nodes, eng_series=engine))

    def test_derive(self):
        x = np.arange(0.0,1.0,0.01)
        y = np.sin(2*2*np.pi*x)+2
        egt_array = np.ma.array(y*440)
        egt = P(name='Eng (2) Gas Temp', array=egt_array)
        takeoffs = buildsection('Takeoff 5 Min Rating', None, 80)

        node = self.node_class()
        node.derive(None, egt, None, None, takeoffs)

        expected = 48

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, expected, delta=1)
        self.assertAlmostEqual(node[0].index, 1, delta=1)
        self.assertEqual(node[0].name, 'Eng (2) Gas Temp Above Normal Max Limit During Takeoff 5 Min Rating Duration')


class TestEngGasTempAboveNormalMaxLimitDuringMaximumContinuousPowerDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngGasTempAboveNormalMaxLimitDuringMaximumContinuousPowerDuration

    def test_can_operate(self):
        nodes = ('Eng (1) Gas Temp', 'Maximum Continous Power')
        engine = A('Engine Series', value='CFM56-5A')
        self.assertFalse(self.node_class.can_operate(nodes, eng_series=engine))
        engine = A('Engine Series', value='CFM56-3')
        self.assertTrue(self.node_class.can_operate(nodes, eng_series=engine))

    def test_derive(self):
        x = np.arange(0.0,1.0,0.01)
        y = np.sin(2*2*np.pi*x)+2
        egt_array = np.ma.array(y*440)
        egt = P(name='Eng (2) Gas Temp', array=egt_array)
        mcp = buildsection('Maximum Continous Power', 55, 80)

        node = self.node_class()
        node.derive(None, egt, None, None, mcp)

        expected = 20

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, expected, delta=1)
        self.assertAlmostEqual(node[0].index, 55, delta=1)
        self.assertEqual(node[0].name, 'Eng (2) Gas Temp Above Normal Max Limit During Maximum Continuous Power Duration')


##############################################################################
# Engine N1


class TestEngN1DuringTaxiMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN1DuringTaxiMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Taxi Out', 'Taxi In', 'Taxiing')]
        self.function = max_value

    def test_derive(self):
        taxi = buildsections('Taxiing', (10, 130), (160, 200))
        taxi_out = buildsections('Taxi Out', (10, 130))
        taxi_in = buildsections('Taxi In', (160, 200))
        eng_n1 = P('Eng (*) N1 Max', array=np.ma.array([80]*10 + [30]*99 + [55] + [98]*10 + [50]*80))
        node = EngN1DuringTaxiMax()
        node.get_derived((eng_n1, taxi_out, taxi_in, taxi))
        self.assertEqual(node[0].value, 55)

    def test_taxiOut_too_short(self):
        taxi = buildsections('Taxiing', (10, 25), (160, 200))
        taxi_out = buildsections('Taxi Out', (10, 25))
        taxi_in = buildsections('Taxi In', (160, 200))
        eng_n1 = P('Eng (*) N1 Max', array=np.ma.array([80]*10 + [30]*99 + [55] + [98]*10 + [50]*80))
        node = EngN1DuringTaxiMax()
        node.get_derived((eng_n1, taxi_out, taxi_in, taxi))
        self.assertEqual(node[0].value, 50)


class TestEngN1DuringTaxiInMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringTaxiInMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Taxi In')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1DuringTaxiOutMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN1DuringTaxiOutMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Taxi Out')]
        self.function = max_value

    def test_derive(self):
        taxi = buildsection('Taxi Out', 10, 130)
        eng_n1 = P('Eng (*) N1 Max', array=np.ma.array([80]*10 + [30]*99 + [55] + [98]*10))
        node = EngN1DuringTaxiOutMax()
        node.get_derived((eng_n1, taxi))
        self.assertEqual(node[0].value, 55)

    def test_index_OOR(self):
        taxi = buildsection('Taxiing', 10, 15)
        taxi.pop() # pop the slice so that get_slices() returns empty array
        eng_n1 = P('Eng (*) N1 Max', array=np.ma.array([80]*10 + [30]*99 + [55] + [98]*10 + [50]*80))
        node = EngN1DuringTaxiOutMax()
        node.get_derived((eng_n1, taxi))
        self.assertEqual(node, [])

class TestEngN1DuringApproachMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringApproachMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN1For5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1For5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN1For5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1MaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngN1For5SecMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN1For5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngAPRDuration(unittest.TestCase):
    def test_can_operate(self):
        combinations = EngAPRDuration.get_operational_combinations()
        self.assertIn(('Mobile', 'FADEC (L) APR Active', 'FADEC (R) APR Active'),
                      combinations)
        self.assertIn(('Mobile', 'Eng (1) ATTCS Armed', 'Eng (1) ATTCS Enabled', 'Eng (1) ATTCS Triggered', 'Eng (2) ATTCS Armed', 'Eng (2) ATTCS Enabled', 'Eng (2) ATTCS Triggered'),
                      combinations)
        self.assertNotIn(('FADEC (L) APR Active', 'FADEC (R) APR Active'),
                         combinations)
        self.assertNotIn(('Mobile', 'FADEC (L) APR Active'),
                         combinations)

    def test_basic(self):
        mobiles = buildsection('Mobile', 0, 14)
        apr_l = P('FADEC (L) APR Active', [0,0,0,0,1,1,1,1,1,1,0,0,0,0])
        apr_r = P('FADEC (R) APR Active', [0,0,0,0,1,1,1,1,1,1,0,0,0,0])
        node = EngAPRDuration()
        node.derive(mobiles, apr_l, apr_r, None, None, None, None, None, None)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 4)
        self.assertEqual(node[0].value, 6)

    def test_gap_and_one_engine(self):
        mobiles = buildsection('Mobile', 0, 630)
        apr_l = P('FADEC (L) APR Active', [0]*10 + [1]*10 + [0]*500 + [1]*10 + [0]*100)
        apr_r = P('FADEC (R) APR Active', [0]*630)
        node = EngAPRDuration()
        node.derive(mobiles, apr_l, apr_r, None, None, None, None, None, None)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 520)

    def test_multiple_two_engines(self):
        mobiles = buildsection('Mobile', 0, 730)
        apr_l = P('FADEC (L) APR Active', [0]*10 + [1]*10 + [0]*600 + [0]*10 + [0]*100)
        apr_r = P('FADEC (R) APR Active', [0]*10 + [0]*10 + [0]*600 + [1]*10 + [0]*100)
        node = EngAPRDuration()
        node.derive(mobiles, apr_l, apr_r, None, None, None, None, None, None)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[1].index, 620)
        self.assertEqual(node[1].value, 10)

    def test_multiple_two_engines_exclude_not_mobile(self):
        mobiles = buildsection('Mobile', 100, 730)
        apr_l = P('FADEC (L) APR Active', [0]*10 + [1]*10 + [0]*600 + [0]*10 + [0]*100)
        apr_r = P('FADEC (R) APR Active', [0]*10 + [0]*10 + [0]*600 + [1]*10 + [0]*100)
        node = EngAPRDuration()
        node.derive(mobiles, apr_l, apr_r, None, None, None, None, None, None)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 620)
        self.assertEqual(node[0].value, 10)

    def test_more_than_5_occurences(self):
        mobiles = buildsection('Mobile', 0, 730 * 6)
        apr_l = P('FADEC (L) APR Active', ([0]*10 + [1]*10 + [0]*600 + [0]*10 + [0]*100)*6)
        apr_r = P('FADEC (R) APR Active', ([0]*10 + [0]*10 + [0]*600 + [1]*10 + [0]*100)*6)
        node = EngAPRDuration()
        node.derive(mobiles, apr_l, apr_r, None, None, None, None, None, None)
        self.assertEqual(len(node), 5)

    def test_attcs(self):
        mobiles = buildsection('Mobile', 0, 14)
        e1_arm = P('Eng (1) ATTCS Armed',        [0,0,0,0,1,1,1,1,1,1,0,0,0,0])
        e1_enabled = P('Eng (1) ATTCS Enabled',  [1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        e1_trigger = P('Eng (1) ATTCS Triggered',[0,0,0,0,0,0,1,1,1,1,1,1,0,0])

        e2_arm = P('Eng (2) ATTCS Armed',        [0,0,0,0,1,1,1,1,1,1,0,0,0,0])
        e2_enabled = P('Eng (2) ATTCS Enabled',  [1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        e2_trigger = P('Eng (2) ATTCS Triggered',[0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        node = EngAPRDuration()
        node.derive(mobiles, None, None, e1_arm, e1_enabled, e1_trigger, e2_arm, e2_enabled, e2_trigger)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 4)


class TestEngN1CyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1CyclesDuringFinalApproach
        self.operational_combinations = [('Eng (*) N1 Avg', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1500To50FtMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1500To50FtMin
        self.operational_combinations = [('Eng (*) N1 Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1For5Sec1000To500FtMin
        self.operational_combinations = [('Eng (*) N1 Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5Sec500To50FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1For5Sec500To50FtMin
        self.operational_combinations = [('Eng (*) N1 Min', 'Altitude AAL For Flight Phases', 'HDF Duration')]
        self.alt_aal = P('Altitude AAL For Flight Phases', array=np.arange(600, 0, -1))
        self.duration = A('HDF Duration', value = 600)

    def test_can_operate(self):
        opts = EngN1For5Sec500To50FtMin.get_operational_combinations()
        self.assertEqual(('Eng (*) N1 Min', 'Altitude AAL For Flight Phases', 'HDF Duration'), opts[0])
        self.assertNotEqual(('Altitude AAL For Flight Phases', 'HDF Duration'), opts[0])

    def test_derive(self):
        low_n1 = np.concatenate((np.arange(50, 20, -1), np.arange(20, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(200) * 50, low_n1, np.ones(340) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 228)
        self.assertEqual(node[0].value, 22)

    def test_derive_exactly_5_seconds(self):
        low_n1 = np.concatenate((np.arange(50, 47, -1), np.arange(48, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(205) * 50, low_n1, np.ones(390) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 205)
        self.assertEqual(node[0].value, 50)

    def test_derive_less_than_5_seconds_begging_of_slice(self):
        low_n1 = np.concatenate((np.arange(50, 46, -1), np.arange(47, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(98) * 50, low_n1, np.ones(500) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 105.0)
        self.assertEqual(node[0].value, 50)

    def test_derive_less_than_5_seconds(self):
        low_n1 = np.concatenate((np.arange(50, 49, -1), np.arange(49, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(295) * 50, low_n1, np.ones(300) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 292)
        self.assertEqual(node[0].value, 50)

    def test_derive_less_than_5_seconds_end_of_slice(self):
        low_n1 = np.concatenate((np.arange(50, 46, -1), np.arange(47, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(546) * 50, low_n1, np.ones(49) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 544)
        self.assertEqual(node[0].value, 50)

    def test_derive_less_than_5_seconds_close_to_end_of_slice(self):
        low_n1 = np.concatenate((np.arange(50, 46, -1), np.arange(47, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(540) * 50, low_n1, np.ones(55) * 50)))

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 541)
        self.assertEqual(node[0].value, 49)

    def test_derive_masked_data(self):
        low_n1 = np.concatenate((np.arange(50, 20, -1), np.arange(20, 50)))
        eng_n1_min_param = P('Eng (*) N1 Min', array=np.concatenate((np.ones(200) * 50, low_n1, np.ones(340) * 50)))
        eng_n1_min_param.array[100:600] = np.ma.masked

        node = self.node_class()
        node.derive(eng_n1_min_param, self.alt_aal, self.duration)

        self.assertEqual(len(node), 0)


class TestEngN1WithThrustReversersInTransitMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1WithThrustReversersInTransitMax
        self.operational_combinations = [('Eng (*) N1 Avg', 'Thrust Reversers', 'Landing')]

    def test_basic(self):
        tr = M(name='Thrust Reversers',
               array=np.ma.array([0,0,0,1,1,2,2,2,2,1,0,0]),
               values_mapping = {0:'Stowed', 1:'In Transit', 2:'Deployed'}
               )
        n1 = P('Eng (*) N1 Avg', array=np.ma.array([50]*6 + [66.0]*1 + [50]*5))
        lands = buildsection('Landing', 2, 11)
        trd = EngN1WithThrustReversersInTransitMax()
        trd.get_derived((n1, tr, lands))
        self.assertAlmostEqual(trd[0].value, 50.0)


class TestEngN1WithThrustReversersDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1WithThrustReversersDeployedMax
        self.operational_combinations = [('Eng (*) N1 Avg', 'Thrust Reversers', 'Landing')]

    def test_basic(self):
        tr = M(name='Thrust Reversers',
               array=np.ma.array([0,0,0,1,1,2,2,2,2,1,0,0]),
               values_mapping = {0:'Stowed', 1:'In Transit', 2:'Deployed'}
               )
        n1 = P('Eng (*) N1 Avg', array=np.ma.array([50]*6 + [66.0]*1 + [50]*5))
        lands = buildsection('Landing', 2, 11)
        trd = EngN1WithThrustReversersDeployedMax()
        trd.get_derived((n1, tr, lands))
        self.assertAlmostEqual(trd[0].value, 66.0)


class TestEngEngStartToN1At60PercentDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = EngEngStartToN1At60PercentDuration
        # Engines N1
        self.eng1 = P('Eng (1) N1', np.ma.array(
            [0, 10, 20, 20, 20, 30, 30, 30, 30, 30,
             59, 61, 61, 40, 40, 40, 65, 75, 85, 85, ]
        ))
        self.eng2 = P('Eng (2) N1', np.ma.array(
            [0, 0, 10, 20, 20, 30, 30, 30, 30, 30,
             59, 61, 61, 40, 40, 40, 65, 75, 85, 85, ]
        ))
        self.eng3 = P('Eng (3) N1', np.ma.array(
            [0, 0, 0, 10, 20, 20, 30, 30, 30, 30,
             59, 61, 61, 40, 40, 40, 65, 75, 85, 85, ]
        ))
        self.eng4 = P('Eng (4) N1', np.ma.array(
            [0, 0, 0, 0, 10, 20, 20, 30, 30, 30,
             59, 61, 61, 40, 40, 40, 65, 75, 85, 85, ]
        ))
        # Engine Starts
        self.engstarts = EngStart(
            'Eng Start',
            items=[
                KeyTimeInstance(1, name='Eng (1) Start'),
                KeyTimeInstance(2, name='Eng (2) Start'),
                KeyTimeInstance(3, name='Eng (3) Start'),
                KeyTimeInstance(4, name='Eng (4) Start'),
            ]
        )
        # Liftoffs
        self.liftoffs = KTI('Liftoff',
                            items=[KeyTimeInstance(16, name='Liftoff'), ])

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 15)
        for opt in opts:
            self.assertTrue('Liftoff' in opt)
            self.assertTrue('Eng Start' in opt)
            self.assertTrue(any_of(('Eng (1) N1', 'Eng (2) N1',
                                    'Eng (3) N1', 'Eng (4) N1'),
                                   opt))

    def test_derive_no_liftoff(self):
        ''' No Liftoff KTI or KTI is empty. '''
        node = self.node_class()
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=self.eng3,
                    eng4=self.eng4, engstarts=self.engstarts, liftoffs=None)
        self.assertEqual(len(node), 0)

        liftoffs = KTI('Liftoff')
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=self.eng3,
                    eng4=self.eng4, engstarts=self.engstarts,
                    liftoffs=liftoffs)
        self.assertEqual(len(node), 0)

    def test_derive_no_eng_start(self):
        ''' No Eng Start KTI or KTI is empty. '''
        node = self.node_class()
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=self.eng3,
                    eng4=self.eng4, engstarts=None, liftoffs=self.liftoffs)
        self.assertEqual(len(node), 0)

        engstarts = EngStart('Eng Start')
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=self.eng3,
                    eng4=self.eng4, engstarts=engstarts,
                    liftoffs=self.liftoffs)
        self.assertEqual(len(node), 0)

    def test_derive_4_engs(self):
        ''' Test 4 Engines'''
        node = self.node_class()
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=self.eng3,
                    eng4=self.eng4, engstarts=self.engstarts,
                    liftoffs=self.liftoffs)
        self.assertEqual(len(node), 4)

        self.assertEqual(
            node[0].name,
            'Eng (1) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[0].index, 1)
        self.assertEqual(node[0].value, 10)

        self.assertEqual(
            node[1].name,
            'Eng (2) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[1].index, 2)
        self.assertEqual(node[1].value, 9)

        self.assertEqual(
            node[2].name,
            'Eng (3) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[2].index, 3)
        self.assertEqual(node[2].value, 8)

        self.assertEqual(
            node[3].name,
            'Eng (4) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[3].index, 4)
        self.assertEqual(node[3].value, 7)

    def test_derive_2_engs(self):
        ''' Test 2 Engines '''
        node = self.node_class()
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=None,
                    eng4=None, engstarts=self.engstarts,
                    liftoffs=self.liftoffs)
        self.assertEqual(len(node), 2)

        self.assertEqual(
            node[0].name,
            'Eng (1) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[0].index, 1)
        self.assertEqual(node[0].value, 10)

        self.assertEqual(
            node[1].name,
            'Eng (2) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[1].index, 2)
        self.assertEqual(node[1].value, 9)

    def test_derive_2_engs_1_start(self):
        ''' Test 2 Engines, but only one starts'''
        engstarts = EngStart(
            'Eng Start',
            items=[KeyTimeInstance(2, name='Eng (2) Start'), ]
        )
        node = self.node_class()
        node.derive(eng1=self.eng1, eng2=self.eng2, eng3=None,
                    eng4=None, engstarts=engstarts,
                    liftoffs=self.liftoffs)
        self.assertEqual(len(node), 1)

        self.assertEqual(
            node[0].name,
            'Eng (2) Eng Start To N1 At 60 Percent Duration'
        )
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 9)


class TestEngN1Below60PercentAfterTouchdownDuration(unittest.TestCase):

    def test_can_operate(self):
        opts = EngN1Below60PercentAfterTouchdownDuration.get_operational_combinations()
        self.assertEqual(('Eng Stop', 'Eng (1) N1', 'Touchdown'), opts[0])
        self.assertEqual(('Eng Stop', 'Eng (2) N1', 'Touchdown'), opts[1])
        self.assertEqual(('Eng Stop', 'Eng (3) N1', 'Touchdown'), opts[2])
        self.assertEqual(('Eng Stop', 'Eng (4) N1', 'Touchdown'), opts[3])
        self.assertTrue(('Eng Stop', 'Eng (1) N1', 'Eng (2) N1', 'Touchdown') in opts)
        self.assertTrue(all(['Touchdown' in avail for avail in opts]))
        self.assertTrue(all(['Eng Stop' in avail for avail in opts]))

    def test_derive_eng_n1_cooldown(self):
        #TODO: Add later if required
        #gnd = S(items=[Section('', slice(10,100))])
        eng_stop = EngStop(items=[KeyTimeInstance(90, 'Eng (1) Stop'),])
        eng = P(array=np.ma.concatenate((np.ones(60) * 100, np.ones(40) * 40))) # idle for 40
        tdwn = KTI(items=[KeyTimeInstance(30), KeyTimeInstance(50)])
        max_dur = EngN1Below60PercentAfterTouchdownDuration()
        max_dur.derive(eng_stop, eng, eng, None, None, tdwn)
        self.assertEqual(max_dur[0].index, 60) # starts at drop below 60
        self.assertEqual(max_dur[0].value, 30) # stops at 90
        self.assertTrue('Eng (1)' in max_dur[0].name)
        # Eng (2) should not be in the results as it did not have an Eng Stop KTI
        ##self.assertTrue('Eng (2)' in max_dur[1].name)
        self.assertEqual(len(max_dur), 1)


class TestEngN1AtTOGADuringTakeoff(unittest.TestCase):

    def test_can_operate(self):
        opts = EngN1AtTOGADuringTakeoff.get_operational_combinations()
        self.assertEqual([('Takeoff And Go Around', 'Eng (*) N1 Min', 'Takeoff')], opts)

    def test_derive_eng_n1_cooldown(self):
        eng_n1_min = P(array=np.ma.arange(10, 20))
        toga = M(array=np.ma.zeros(10), values_mapping={0: '-', 1:'TOGA'})
        toga.array[3] = 1
        toff = buildsection('Takeoff', 2,6)
        n1_toga = EngN1AtTOGADuringTakeoff()
        n1_toga.derive(toga=toga,
                       eng_n1=eng_n1_min,
                       takeoff=toff)
        self.assertEqual(n1_toga[0].value, 13)
        self.assertEqual(n1_toga[0].index, 3)


class TestEngN154to72PercentWithThrustReversersDeployedDurationMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN154to72PercentWithThrustReversersDeployedDurationMax

    def test_can_operate(self):
        eng_series = A(name='Engine Series', value='Tay 620')
        expected = [('Eng (1) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Thrust Reversers'),
                    ('Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers')]
        for combination in expected:
            self.assertTrue(self.node_class().can_operate(combination, eng_series))
        eng_series.value = 'Tay 611'
        for combination in expected:
            self.assertFalse(self.node_class().can_operate(combination, eng_series))

    def test_derive(self):
        values_mapping = {
            0: 'Stowed',
            1: 'In Transit',
            2: 'Deployed',
        }
        n1_array = 30*(2+np.sin(np.arange(0, 12.6, 0.1)))
        eng_1 = P(name='Eng (1) N1', array=np.ma.array(n1_array))
        thrust_reversers_array = np.ma.zeros(126)
        thrust_reversers_array[55:94] = 2
        thrust_reversers = M('Thrust Reversers', array=thrust_reversers_array, values_mapping=values_mapping)

        node = self.node_class()
        eng_series = A(name='Engine Series', value='Tay 620')
        node.derive(eng_1, None, None, None, thrust_reversers, eng_series)

        self.assertEqual(node[0].name, 'Eng (1) N1 54 To 72 Percent With Thrust Reversers Deployed Duration Max')
        self.assertEqual(node[0].index, 61)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(len(node), 1)


class TestEngN1ExceededN1RedlineDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN1ExceededN1RedlineDuration
        array = np.ma.concatenate((np.zeros(5), np.arange(10), np.ones(10) * 10))
        self.array = np.ma.concatenate((array, array[::-1]))

    def test_derive(self):
        n1_1 = P(name='Eng (1) N1', array=self.array.copy())
        n1_1_red = P(name='Eng (1) N1 Redline', array=self.array.copy()+20)
        n1_2 = P(name='Eng (2) N1', array=self.array.copy())
        n1_2_red = P(name='Eng (2) N1 Redline', array=self.array.copy()+20)
        n1_2.array[13:17] = 35

        node = self.node_class()
        node.derive(n1_1, n1_1_red, n1_2, n1_2_red, *[None]*4)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].name, 'Eng N1 Exceeded N1 Redline Duration')


##############################################################################
# Engine N2


class TestEngN2DuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringTaxiMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN2For5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN2For5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN2DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2For5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN2For5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2MaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Maximum Continuous Power')]
        self.function = max_value




class TestEngN2For5SecMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN2For5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2CyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN2CyclesDuringFinalApproach
        self.operational_combinations = [('Eng (*) N2 Avg', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2ExceededN2RedlineDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN2ExceededN2RedlineDuration
        array = np.ma.concatenate((np.zeros(5), np.arange(10), np.ones(10) * 10))
        self.array = np.ma.concatenate((array, array[::-1]))

    def test_derive(self):
        n2_1 = P(name='Eng (1) N2', array=self.array.copy())
        n2_1_red = P(name='Eng (1) N2 Redline', array=self.array.copy()+20)
        n2_2 = P(name='Eng (2) N2', array=self.array.copy())
        n2_2_red = P(name='Eng (2) N2 Redline', array=self.array.copy()+20)
        n2_2.array[13:17] = 35

        node = self.node_class()
        node.derive(n2_1, n2_1_red, n2_2, n2_2_red, *[None]*4)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].name, 'Eng N2 Exceeded N2 Redline Duration')


##############################################################################
# Engine N3


class TestEngN3DuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringTaxiMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN3For5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN3For5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN3DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3For5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN3For5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3MaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngN3For5SecMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngN3For5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3ExceededN3RedlineDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN3ExceededN3RedlineDuration
        array = np.ma.concatenate((np.zeros(5), np.arange(10), np.ones(10) * 10))
        self.array = np.ma.concatenate((array, array[::-1]))

    def test_derive(self):
        n3_1 = P(name='Eng (1) N3', array=self.array.copy())
        n3_1_red = P(name='Eng (1) N3 Redline', array=self.array.copy()+20)
        n3_2 = P(name='Eng (2) N3', array=self.array.copy())
        n3_2_red = P(name='Eng (2) N3 Redline', array=self.array.copy()+20)
        n3_2.array[13:17] = 35

        node = self.node_class()
        node.derive(n3_1, n3_1_red, n3_2, n3_2_red, *[None]*4)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].name, 'Eng N3 Exceeded N3 Redline Duration')


##############################################################################
# Engine Np


class TestEngNpDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringTaxiMax
        self.operational_combinations = [('Eng (*) Np Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpDuringTakeoff5MinRatingMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngNpDuringTakeoff5MinRatingMax
        self.operational_combinations = [
            ('Eng (*) Np Max', 'Takeoff 5 Min Rating'),
            ('Eng (*) Np Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngNpFor5SecDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngNpFor5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Np Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngNpDuringClimbMin(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringClimbMin
        self.operational_combinations = [('Eng (*) Np Min', 'Climbing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Np Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpFor5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngNpFor5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Np Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Np Max', 'Maximum Continuous Power')]
        self.function = max_value


class TestEngNpForXSecMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngNpForXSecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Np Max', 'Maximum Continuous Power')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNp82To90PercentDurationMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngNp82To90PercentDurationMax

    def test_can_operate(self):
        ac_series = A(name='Series', value='Jetstream 41')
        expected = [('Eng (1) Np', 'Eng (2) Np'),]
        for combination in expected:
            self.assertTrue(self.node_class().can_operate(combination, ac_series))
        ac_series.value = 'Jetstream'
        for combination in expected:
            self.assertFalse(self.node_class().can_operate(combination, ac_series))

    def test_derive(self):
        eng_1 = P(name='Eng (1) Np', array=np.ma.arange(80, 92))
        node = self.node_class()
        node.derive(eng_1, eng_1) # Intentional duplication of data

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].name, 'Eng (1) Np 82 To 90 Percent Duration Max')
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 9)
        self.assertEqual(node[1].name, 'Eng (2) Np 82 To 90 Percent Duration Max')
        self.assertEqual(node[1].index, 2)
        self.assertEqual(node[1].value, 9)

##############################################################################
# Engine Throttles


class TestThrottleReductionToTouchdownDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrottleReductionToTouchdownDuration
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [
            ('Throttle Levers', 'Eng (*) N1 Avg', 'Landing', 'Touchdown',
             'Frame')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Oil Pressure


class TestEngOilPressMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngOilPressMax
        self.operational_combinations = [
            ('Eng (*) Oil Press Max',),
            ('Taxi Out', 'Eng (*) Oil Press Max',),
            ('First Eng Start Before Liftoff', 'Eng (*) Oil Press Max'),
            ('Eng (*) Oil Press Max', 'Touchdown'),
            ('Taxi Out', 'First Eng Start Before Liftoff', 'Eng (*) Oil Press Max'),
            ('Taxi Out', 'Eng (*) Oil Press Max', 'Touchdown'),
            ('First Eng Start Before Liftoff', 'Eng (*) Oil Press Max', 'Touchdown'),
            ('Taxi Out', 'First Eng Start Before Liftoff', 'Eng (*) Oil Press Max', 'Touchdown'),
        ]
        self.oil_press = P(name='Eng (*) Oil Press Max',
                           array=np.ma.array([
                               10, 40, 50, 60, 70,
                               160, 200, 70, 20, 40,
                               160, 140, 170, 210, 30,
                           ]))

    def test_derive_single_ktis(self):
        start_kti = KTI(name='First Eng Start Before Liftoff',
                        items=[KeyTimeInstance(name='First Eng Start Before Liftoff',
                                               index=2),])

        end_kti = KTI(name='Touchdown',
                      items=[KeyTimeInstance(name='Touchdown',
                                             index=13),])
        node = self.node_class()
        node.derive(self.oil_press, None, start_kti, end_kti)

        self.assertAlmostEqual(int(node[0].value), 200)
        self.assertAlmostEqual(int(node[0].index), 6)

    def test_derive_multi_ktis(self):
        start_kti = KTI(name='First Eng Start Before Liftoff',
                        items=[
                            KeyTimeInstance(name='First Eng Start Before Liftoff',index=1),
                            KeyTimeInstance(name='First Eng Start Before Liftoff',index=3),
                        ])

        end_kti = KTI(name='Touchdown',
                      items=[
                          KeyTimeInstance(name='Touchdown', index=13),
                          KeyTimeInstance(name='Touchdown', index=15),
                      ])
        node = self.node_class()
        node.derive(self.oil_press, None, start_kti, end_kti)

        self.assertAlmostEqual(int(node[0].value), 210)
        self.assertAlmostEqual(int(node[0].index), 13)

    def test_derive_only_array(self):
        node = self.node_class()
        node.derive(self.oil_press, None, None, None)

        self.assertAlmostEqual(int(node[0].value), 210)
        self.assertAlmostEqual(int(node[0].index), 13)

    def test_derive_taxi_no_ktis(self):
        node = self.node_class()
        node.derive(
            self.oil_press,
            buildsection('Taxi Out', 14, len(self.oil_press.array)),
            None,
            None,
        )

        self.assertAlmostEqual(int(node[0].value), 30)
        self.assertAlmostEqual(int(node[0].index), 14)

    def test_derive_taxi_and_ktis(self):
        start_kti = KTI(name='First Eng Start Before Liftoff',
                        items=[KeyTimeInstance(name='First Eng Start Before Liftoff',
                                               index=2),])
        end_kti = KTI(name='Touchdown',
                      items=[KeyTimeInstance(name='Touchdown',
                                             index=13),])
        node = self.node_class()
        node.derive(
            self.oil_press,
            buildsection('Taxi Out', 7, 13),
            start_kti,
            end_kti,
        )

        self.assertAlmostEqual(int(node[0].value), 170)
        self.assertAlmostEqual(int(node[0].index), 12)


class TestEngOilPressFor60SecDuringCruiseMax(unittest.TestCase):

    def test_can_operate(self):
        node = EngOilPressFor60SecDuringCruiseMax
        self.assertEqual(node.get_operational_combinations(),
                         [('Eng (*) Oil Press Max', 'Cruise')])

    def test_derive(self):
        press = P('Eng (*) Oil Press Max', frequency=0.5,
                  array=np.ma.array([22]*60 + [52]*39 + [17]*60 + [100]*300))
        cruise = buildsection('Cruise', 66, 126)
        under_pressure = EngOilPressFor60SecDuringCruiseMax()
        under_pressure.derive(press, cruise)
        self.assertEqual(len(under_pressure), 1)
        self.assertEqual(under_pressure[0].value, 52)


class TestEngOilPressMin(unittest.TestCase):

    def setUp(self):
        self.node_class = EngOilPressMin
        self.operational_combinations = [('Eng (*) Oil Press Min', 'Airborne')]

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive(self):
        oil_p = P(
            name='Eng (*) Oil Press Min',
            array=np.ma.array(data=[50, 50, 50, 10, 10, 50, 50, 50], dtype=float),
        )
        airborne = buildsection('Airborne', 1, 6)
        node = self.node_class()
        node.derive(oil_p, airborne)
        self.assertEqual(node, KPV('Eng Oil Press Min',
                                   items=[KeyPointValue(
                    index=3.0, value=10.0,
                    name='Eng Oil Press Min')]))

    def test_zero(self):
        oil_p = P(
            name='Eng (*) Oil Press Min',
            array=np.ma.array(data=[50, 50, 50, 0, 0, 50, 50, 50], dtype=float),
        )
        airborne = buildsection('Airborne', 1, 6)
        node = self.node_class()
        node.derive(oil_p, airborne)
        self.assertEqual(node, KPV('Eng Oil Press Min',
                                   items=[KeyPointValue(
                    index=3.0, value=0.0,
                    name='Eng Oil Press Min')]))

    def test_single_zero(self):
        oil_p = P(
            name='Eng (*) Oil Press Min',
            array=np.ma.array(data=[50, 50, 50, 50, 0, 50, 45, 50], dtype=float),
        )
        airborne = buildsection('Airborne', 1, 6)
        node = self.node_class()
        node.derive(oil_p, airborne)
        self.assertEqual(node, KPV('Eng Oil Press Min',
                                   items=[KeyPointValue(
                    index=6.0, value=45.0,
                    name='Eng Oil Press Min')]))

    def test_all_zero(self):
        oil_p = P(
            name='Eng (*) Oil Press Min',
            array=np.ma.array(data=[0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        )
        airborne = buildsection('Airborne', 1, 6)
        node = self.node_class()
        node.derive(oil_p, airborne)
        self.assertEqual(node, [])


class TestEngOilPressWarningDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngOilPressWarningDuration
        self.airborne = buildsection('Airborne', 2, 17)
        self.oil_press_warn = M('Eng (*) Oil Press Warning',
                                np.ma.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             1, 1, 1, 1, 1, 0, 0, 0, 0, 1]),
                                values_mapping={0:'-', 1:'Warning'})

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Eng Oil Press Warning Duration')
        self.assertEqual(node.units, ut.SECOND)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts),1)
        self.assertIn('Eng (*) Oil Press Warning', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        node = self.node_class()
        node.derive(self.oil_press_warn, self.airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[0].index, 10)


class TestEngOilPressLowRedlineExceededDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = EngOilPressLowRedlineExceededDuration
        self.values_mapping = {0:'-', 1:'Exceeded'}
        self.press_low_1 = M('Eng (1) Oil Press Low Redline Exceeded',
                             np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
                             values_mapping=self.values_mapping)
        self.press_low_2 = M('Eng (2) Oil Press Low Redline Exceeded',
                             np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
                             values_mapping=self.values_mapping)

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'Eng Oil Press Low Redline Exceeded Duration')
        self.assertEqual(node.units, ut.SECOND)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertIn('Eng (1) Oil Press Low Redline Exceeded', opts[0])
        self.assertIn('Eng (2) Oil Press Low Redline Exceeded', opts[0])

    def test_derive_overlapping(self):
        node = self.node_class()
        node.derive(self.press_low_1, self.press_low_2)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 7)
        self.assertEqual(node[0].index, 10)

    def test_derive_no_warnings(self):
        press_low_1 = M('Eng (1) Oil Press Low Redline Exceeded',
                        np.ma.array([0]*20),
                        values_mapping=self.values_mapping)
        press_low_2 = M('Eng (2) Oil Press Low Redline Exceeded',
                        np.ma.array([0]*20),
                        values_mapping=self.values_mapping)

        node = self.node_class()
        node.derive(press_low_1, press_low_2)

        self.assertEqual(len(node), 0)


##############################################################################
# Engine Oil Quantity


class TestEngOilQtyMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilQtyMax
        self.operational_combinations = [('Eng (*) Oil Qty Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilQtyMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilQtyMin
        self.operational_combinations = [('Eng (*) Oil Qty Min', 'Airborne')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilQtyDuringTaxiInMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = EngOilQtyDuringTaxiInMax
        self.operational_combinations = [('Eng (1) Oil Qty', 'Taxi In')]
        self.function = max_value

    def test_derive(self):
        oil_qty = P(
            name='Eng (1) Oil Qty',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        taxi_in = S(items=[Section('Taxi In', slice(3, 9), 3, 9)])
        node = self.node_class()
        node.derive(oil_qty, None, None, None, taxi_in)
        self.assertEqual(
            node,
            KPV('Eng (1) Oil Qty During Taxi In Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Eng (1) Oil Qty During Taxi In Max')]))

    def test_derive_from_hdf(self):
        [oil], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, '757-3A-001.hdf5'),
            ['Eng (1) Oil Qty'], slice(21722, 21936), 'Taxi In')
        node = self.node_class()
        node.derive(oil, None, None, None, phase)
        self.assertEqual(
            node,
            KPV('Eng (1) Oil Qty During Taxi In Max',
                items=[KeyPointValue(
                    index=21725.0, value=16.015625,
                    name='Eng (1) Oil Qty During Taxi In Max')]))


class TestEngOilQtyDuringTaxiOutMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = EngOilQtyDuringTaxiOutMax
        self.operational_combinations = [('Eng (1) Oil Qty', 'Taxi Out')]
        self.function = max_value

    def test_derive(self):
        oil_qty = P(
            name='Eng (1) Oil Qty',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        taxi_out = S(items=[Section('Taxi Out', slice(3, 9), 3, 9)])
        node = self.node_class()
        node.derive(oil_qty, None, None, None, taxi_out)
        self.assertEqual(
            node,
            KPV('Eng (1) Oil Qty During Taxi Out Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Eng (1) Oil Qty During Taxi Out Max')]))

##############################################################################
# Engine Oil Temperature


class TestEngOilTempMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilTempMax
        self.operational_combinations = [('Eng (*) Oil Temp Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngOilTempForXMinMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngOilTempForXMinMax
        self.operational_combinations = [('Eng (*) Oil Temp Max', )]

    def test_derive_real_case(self):
        array = np.ma.array(
            [67.0,79,81,82,84,85,87,88,90,90,92,93,93,94,
             94,95,97,103,109,112,115,118,119,121,121,
             123,123,124,124,125,125,124,122,121,121,120,
             120]+[119]*34+[117]*17+[115]*98+[112]*5+[106]*65+[103]*80)
        array = np.repeat(array, 10)
        oil_temp = P(name='Eng (*) Oil Temp Max', array=array)
        oil_temp.array[-3:]=np.ma.masked
        node = EngOilTempForXMinMax()
        node.derive(oil_temp)
        self.assertEqual(len(node), 3)
        kpv_15 = next(x for x in node if
                      x.name == 'Eng Oil Temp For 15 Min Max')
        self.assertEqual(kpv_15.index, 200, 115)
        kpv_20 = next(x for x in node if
                      x.name == 'Eng Oil Temp For 20 Min Max')
        self.assertEqual(kpv_20.index, 200, 115)
        kpv_45 = next(x for x in node if
                      x.name == 'Eng Oil Temp For 45 Min Max')
        self.assertEqual(kpv_45.index, 170, 103)

    def test_derive_all_oil_data_masked(self):
        # This has been a specific problem, hence this test.
        oil_temp = P(
            name='Eng (*) Oil Temp Max',
            array=np.ma.array(data=range(123, 128), dtype=float, mask=True),
        )
        node = EngOilTempForXMinMax()
        node.derive(oil_temp)
        self.assertEqual(node, KPV('Eng Oil Temp For X Min Max', items=[]))


##############################################################################
# Engine Torque


class TestEngTorqueDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueDuringTaxiMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueDuringTakeoff5MinRatingMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngTorqueDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Takeoff 5 Min Rating')]
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.function = max_value

    def test_can_operate_heli(self):
        operational_combinations = [('Eng (*) Torque Max', 'Takeoff 5 Min Rating', 'All Engines Operative')]
        kwargs = {'ac_type': helicopter}
        combinations = list(map(set, self.node_class.get_operational_combinations(**kwargs)))
        for combination in map(set, operational_combinations):
            self.assertIn(combination, combinations)

    def test_derive(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))
        ratings = buildsection('Takeoff 5 Min Rating', 1, 28)

        node = self.node_class()
        node.derive(eng, ratings, None)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 27)
        self.assertEqual(node[0].value, 75)

    def test_derive_heli(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))

        ratings = buildsection('Takeoff 5 Min Rating', 1, 28)
        all_eng = M('All Engines Operative', np.ma.array([1]*14 + [0]*16), values_mapping={0:'-', 1:'AEO'})

        node = self.node_class()
        node.derive(eng, ratings, all_eng)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 73)


class TestEngFor5SecTorqueDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTorqueFor5SecDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Takeoff 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTorqueDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueFor5SecDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTorqueFor5SecDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Go Around 5 Min Rating')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngTorqueDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Maximum Continuous Power')]
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.function = max_value

    def test_can_operate_heli(self):
        operational_combinations = [('Eng (*) Torque Max', 'Maximum Continuous Power', 'All Engines Operative')]
        kwargs = {'ac_type': helicopter}
        combinations = list(map(set, self.node_class.get_operational_combinations(**kwargs)))
        for combination in map(set, operational_combinations):
            self.assertIn(combination, combinations)

    def test_derive(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))
        mcp = buildsection('Maximum Continuous Power', 1, 28)

        node = self.node_class()
        node.derive(eng, mcp, None)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 27)
        self.assertEqual(node[0].value, 75)

    def test_derive_heli(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))

        mcp = buildsection('Maximum Continuous Power', 1, 28)
        all_eng = M('All Engines Operative', np.ma.array([1]*14 + [0]*16), values_mapping={0:'-', 1:'AEO'})

        node = self.node_class()
        node.derive(eng, mcp, all_eng)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 73)


class TestEngTorqueFor5SecMaximumContinuousPowerMax(unittest.TestCase, CreateKPVsWithinSlicesSecondWindowTest):

    def setUp(self):
        self.node_class = EngTorqueFor5SecDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Maximum Continuous Power')]
        self.function = max_value
        self.duration = 5

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorque500To50FtMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorque500To50FtMin
        self.operational_combinations = [('Eng (*) Torque Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTorqueWhileDescendingMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueWhileDescendingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Descending')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque7FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTorque7FtToTouchdownMax
        self.touchdowns = KTI(name='Touchdown',
                              items=[KeyTimeInstance(name='Touchdown',
                                                     index=5),])

    def test_can_operate(self):
        prop = A('Engine Propulsion', 'PROP')
        opts = self.node_class.get_operational_combinations(eng_type=prop)

        self.assertEqual(len(opts), 1)
        self.assertIn('Eng (*) Torque Max', opts[0])
        self.assertIn('Altitude AAL For Flight Phases', opts[0])
        self.assertIn('Touchdown', opts[0])

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Eng Torque 7 Ft To Touchdown Max')
        self.assertEqual(node.units, ut.PERCENT)

    def test_derive(self):
        torque = P('Eng (*) Torque Max',
                   np.ma.array([80, 80, 81, 85, 90, 80, 78]))
        alt_aal = P('Altitude AAL For Flight Phases',
                    np.ma.array([100, 70, 45, 20, 6, 0, 0]))

        node = self.node_class()
        node.derive(torque, alt_aal, self.touchdowns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 4)
        self.assertEqual(node[0].value, 90)


class TestEngTorqueWithin1SecOfTouchdownMax(unittest.TestCase):
    def setUp(self):
        self.node_class = EngTorqueWithin1SecOfTouchdownMax
        self.touchdown = KTI(name='Touchdown',
                             items=[KeyTimeInstance(name='Touchdown',
                                                    index=5),])
    def test_basic(self):
        torque=P('Eng (*) Torque Max',
                 np.ma.arange(10,0,-1))

        node = self.node_class()
        node.derive(self.touchdown,torque)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(node[0].index, 4)

    def test_half_hertz(self):
        torque=P('Eng (*) Torque Max',
                 np.ma.arange(10,0,-1),
                 frequency=0.5)

        node = self.node_class()
        node.derive(self.touchdown,torque)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(node[0].index, 4)

    def test_hi_frequency(self):
        touchdown = KTI(name='Touchdown',
                        items=[KeyTimeInstance(name='Touchdown',
                                                    index=13),])
        torque=P('Eng (*) Torque Max',
                 np.ma.arange(30,0,-1),
                 frequency=8)

        node = self.node_class()
        node.derive(touchdown, torque)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 18)
        self.assertEqual(node[0].index, 12)


class TestTorqueAsymmetryWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = TorqueAsymmetryWhileAirborneMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Torque Asymmetry', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 400)
        torque = P(
            name='Torque Asymmetry',
            array=np.ma.abs(x*np.sin(x)),
        )
        name = 'Airborne'
        section = Section(name, slice(50, 350), 50, 350)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(torque, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 318)
        self.assertAlmostEqual(node[0].value, 7.916, places=3)


##############################################################################
# Engine Vibration


class TestEngVibN1Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN1Max
        self.operational_combinations = [('Eng (*) Vib N1 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibN2Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN2Max
        self.operational_combinations = [('Eng (*) Vib N2 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibN3Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN3Max
        self.operational_combinations = [('Eng (*) Vib N3 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibAMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibAMax
        self.operational_combinations = [('Eng (*) Vib A Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibBMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibBMax
        self.operational_combinations = [('Eng (*) Vib B Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibCMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibCMax
        self.operational_combinations = [('Eng (*) Vib C Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibBroadbandMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngVibBroadbandMax
        self.operational_combinations = [('Eng (*) Vib Broadband Max',)]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibNpMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibNpMax
        self.operational_combinations = [('Eng (*) Vib Np Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')

##############################################################################
# Engine Derate calculations

class TestEngN1TakeoffDerate(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN1TakeoffDerate

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([]))
        self.assertFalse(self.node_class.can_operate(['Eng (*) N1 Avg', 'TAT', 'Mach', 'SAGE Takeoff']))
        self.assertFalse(self.node_class.can_operate(['Eng (*) N1 Avg', 'TAT', 'Mach', 'SAGE Takeoff', 'Engine Series'],
                                                     A('Engine Series', value='RB199')))
        self.assertTrue(self.node_class.can_operate(['Eng (*) N1 Avg', 'TAT', 'Mach', 'SAGE Takeoff', 'Engine Series'],
                                                    A('Engine Series', value='CFM56-5B')))

    def test_derive(self):
        toff =  KTI('SAGE Takeoff',
                    items=[KeyTimeInstance(5, 'SAGE Takeoff')])
        eng_n1 = P('Eng (*) N1 Avg', array=np.ma.array([87.25]*10))
        tat = P('TAT', array=np.ma.array([39.5]*10))
        mach = P('Mach', array=np.ma.array([0.252]*10))
        node = self.node_class()
        node.derive(eng_n1, tat, mach, toff)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 16.24, places=2)

    def test_out_of_range(self):
        toff =  KTI('SAGE Takeoff',
                    items=[KeyTimeInstance(5, 'SAGE Takeoff')])
        eng_n1 = P('Eng (*) N1 Avg', array=np.ma.array([84.0]*10))
        tat = P('TAT', array=np.ma.array([39.5]*10))
        node = self.node_class()
        node.derive(eng_n1, tat, toff)
        self.assertEqual(len(node), 0)

    def test_overspeed(self):
        toff =  KTI('SAGE Takeoff',
                    items=[KeyTimeInstance(5, 'SAGE Takeoff')])
        eng_n1 = P('Eng (*) N1 Avg', array=np.ma.array([87.25]*10))
        tat = P('TAT', array=np.ma.array([39.5]*10))
        mach = P('Mach', array=np.ma.array([0.32]*10))
        node = self.node_class()
        node.derive(eng_n1, tat, mach, toff)
        self.assertEqual(len(node), 0)


class TestEngThrustTakeoffDerate(unittest.TestCase):

    def setUp(self):
        self.node_class = EngThrustTakeoffDerate

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Mach', 'Eng N1 Takeoff Derate')])

    def test_derive(self):
        mach = P('Mach', array=np.ma.array([0.25]*10))
        eng_dr = KPV('Eng N1 Takeoff Derate',
                     items=[KeyPointValue(index=5, value=14)])
        node = self.node_class()
        node.derive(mach, eng_dr)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 13.15, places=2)


    def test_derive_precision(self):
        mach = P('Mach', array=np.ma.array([0.252]*10))
        eng_dr = KPV('Eng N1 Takeoff Derate',
                     items=[KeyPointValue(index=5, value=16.238)])
        node = self.node_class()
        node.derive(mach, eng_dr)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].value, 21.1, places=1)


class TestEngTakeoffFlexTemp(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTakeoffFlexTemp

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(['Flex Temp', 'SAGE Takeoff']))
        self.assertTrue(self.node_class.can_operate(['Eng (1) Flex Temp', 'Eng (2) Flex Temp', 'SAGE Takeoff']))

    def test_derive(self):
        toff =  KTI('SAGE Takeoff',
                    items=[KeyTimeInstance(5, 'SAGE Takeoff')])
        eng_flex = P('Flex Temp', array=np.ma.arange(10, dtype=float))
        node = self.node_class()
        node.derive(toff, eng_flex, None, None)
        self.assertEqual(node[0].value, 5.0)

    def test_derive_two_sources(self):
        toff =  KTI('SAGE Takeoff',
                    items=[KeyTimeInstance(5, 'SAGE Takeoff')])
        eng_flex_1 = P('Eng (1) Flex Temp', array=np.ma.arange(10, dtype=float))
        eng_flex_2 = P('Eng (2) Flex Temp', array=np.ma.arange(10, dtype=float) + 1.0)
        node = self.node_class()
        node.derive(toff, None, eng_flex_1, eng_flex_2)
        self.assertEqual(node[0].value, 5.5)


##############################################################################
# Engine: Warnings

# Chip Detection

class TestEngChipDetectorWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngChipDetectorWarningDuration
        self.values_mapping = {0: '-', 1: 'Chip Detected'}
        self.operational_combinations = [
            ('Eng (1) Chip Detector', 'Eng (*) Any Running'),
            ('Eng (2) Chip Detector', 'Eng (*) Any Running'),
            ('Eng (1) Chip Detector (1)', 'Eng (*) Any Running'),
            ('Eng (1) Chip Detector (1)', 'Eng (2) Chip Detector (1)', 'Eng (1) Chip Detector (2)', 'Eng (2) Chip Detector (2)', 'Eng (*) Any Running'),
        ]

    def test_derive_basic(self):
        array = np.ma.array([0] * 7 + [1] * 3 + [0] * 6)
        eng_1_chip = M('Eng (1) Chip Detector', array, values_mapping=self.values_mapping)
        eng_2_chip = M('Eng (2) Chip Detector', np.roll(array, 2), values_mapping=self.values_mapping)
        eng_2_chip.array[0] = 'Chip Detected'
        eng_1_chip_1 = M('Eng (1) Chip Detector (1)', np.roll(array, 4), values_mapping=self.values_mapping)

        running = M('Eng (*) Any Running', np.ma.zeros(16), values_mapping={0: 'Not Running', 1: 'Running'})
        running.array[3:13] = 'Running'
        running.array.mask = np.ma.getmaskarray(running.array)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=7, value=6),
        ])
        node = self.node_class()
        node.derive(eng_1_chip, eng_2_chip, eng_1_chip_1, None, None, None, running)
        self.assertEqual(node, expected)


class TestGearboxChipDetectorWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearboxChipDetectorWarningDuration
        self.values_mapping = {0: '-', 1: 'Chip Detected'}
        self.operational_combinations = [
            ('EGB (1) Chip Detector', 'Eng (*) Any Running'),
            ('MGB Chip Detector', 'Eng (*) Any Running'),
            ('CGB Chip Detector', 'Eng (*) Any Running'),
            ('IGB Chip Detector', 'IGB Chip Detector', 'Eng (*) Any Running'),
        ]

    def test_derive_basic(self):
        array = np.ma.array([0] * 7 + [1] * 3 + [0] * 6)
        egb_1_chip = M('EGB (1) Chip Detector', array, values_mapping=self.values_mapping)
        mgb_chip = M('MGB Chip Detector', np.roll(array, 2), values_mapping=self.values_mapping)
        mgb_chip.array[0] = 'Chip Detected'
        cgb_chip = M('CGB Chip Detector', np.roll(array, 4), values_mapping=self.values_mapping)

        running = M('Eng (*) Any Running', np.ma.zeros(16), values_mapping={0: 'Not Running', 1: 'Running'})
        running.array[3:13] = 'Running'
        running.array.mask = np.ma.getmaskarray(running.array)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=7, value=6),
        ])
        node = self.node_class()
        node.derive(egb_1_chip, None, mgb_chip, None, None, None, None, None, None, None, cgb_chip, None, running)
        self.assertEqual(node, expected)



##############################################################################


class TestEventMarkerPressed(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLoss1000To2000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLoss1000To2000Ft
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Descend For Flight Phases',
            'Altitude AAL For Flight Phases',
            'Climb',
        )]

    def test_basic(self):
        vs = P(
            name='Descend For Flight Phases',
            array=np.ma.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                               0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            frequency=2.0,
            offset=0.0,
        )
        alt = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([0, 4, 1000, 1600, 1900, 2100, 1900, 1950, 2100, 1600, 1150, 0]),
            frequency=1.0,
            offset=0.5,
        )
        climb = buildsection('Climb', 2, 6)
        ht_loss = self.node_class()
        ht_loss.get_derived((vs, alt, climb))
        self.assertEqual(len(ht_loss), 1)
        self.assertEqual(ht_loss[0].value, 1)
        self.assertEqual(ht_loss[0].index, 6)


class TestHeightLoss35To1000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLoss35To1000Ft
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Descend For Flight Phases',
            'Altitude AAL For Flight Phases',
            'Initial Climb',
        )]

    def test_basic(self):
        vs = P(
            name='Descend For Flight Phases',
            array=np.ma.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                               0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            frequency=2.0,
            offset=0.0,
        )
        alt = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([0, 4, 40, 150, 600, 1100, 900, 950, 1100, 600, 150, 0]),
            frequency=1.0,
            offset=0.5,
        )
        climb = buildsection('Initial Climb', 1.8, 5)
        ht_loss = self.node_class()
        ht_loss.get_derived((vs, alt, climb))
        self.assertEqual(len(ht_loss), 1)
        self.assertEqual(ht_loss[0].value, 1)
        self.assertEqual(ht_loss[0].index, 6)


class TestHeightLossLiftoffTo35Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLossLiftoffTo35Ft
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Vertical Speed Inertial',
            'Altitude AAL For Flight Phases',
        )]

    def test_basic(self):
        vs = P(
            name='Vertical Speed Inertial',
            array=np.ma.array([0.0, 0, 1, 2, 1, 0, -1, -2, 0, 4]),
            frequency=2.0,
            offset=0.0,
        )
        alt = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([0.0, 0, 4, 15, 40]),
            frequency=1.0,
            offset=0.5,
        )
        ht_loss = HeightLossLiftoffTo35Ft()
        ht_loss.get_derived((vs, alt))
        self.assertEqual(ht_loss[0].value, 0.75)


class TestHeightOfBouncedLanding(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeAtTouchdown
        self.operational_combinations = [
            ('Latitude', 'Touchdown'),
            ('Touchdown', 'AFR Landing Airport'),
            ('Touchdown', 'AFR Landing Runway'),
            ('Touchdown', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport'),
            ('Latitude', 'Touchdown', 'AFR Landing Runway'),
            ('Latitude', 'Touchdown', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Touchdown', 'AFR Landing Airport', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Latitude (Coarse)')
        ]

    def test_derive_with_latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        tdwns = KTI(name='Touchdown')
        afr_land_rwy = None
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_rwy, afr_land_apt, lat_c, aeroplane)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, tdwns)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_land_rwy(self):
        lat = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = A(name='AFR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, lat_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_land_apt(self):
        lat = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = None
        afr_land_apt = A(name='AFR Landing Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

class TestLatitudeAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeAtLiftoff
        self.operational_combinations = [
            ('Latitude', 'Liftoff'),
            ('Liftoff', 'AFR Takeoff Airport'),
            ('Liftoff', 'AFR Takeoff Runway'),
            ('Liftoff', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Runway'),
            ('Latitude', 'Liftoff', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Liftoff', 'AFR Takeoff Airport', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
        ]

    def test_derive_with_latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_rwy, afr_toff_apt, None, aeroplane)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, liftoffs)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_toff_rwy(self):
        lat = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = A(name='AFR Takeoff Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, lat_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_toff_apt(self):
        lat = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = A(name='AFR Takeoff Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLatitudeSmoothedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeSmoothedAtTouchdown
        self.operational_combinations = [('Latitude Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLatitudeSmoothedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeSmoothedAtLiftoff
        self.operational_combinations = [('Latitude Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLatitudeAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Latitude Prepared', 'Lowest Altitude During Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeAtTouchdown
        self.operational_combinations = [
            ('Longitude', 'Touchdown'),
            ('Touchdown', 'AFR Landing Airport'),
            ('Touchdown', 'AFR Landing Runway'),
            ('Touchdown', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport'),
            ('Longitude', 'Touchdown', 'AFR Landing Runway'),
            ('Longitude', 'Touchdown', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Touchdown', 'AFR Landing Airport', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Longitude (Coarse)')
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Latitude')
        lon.array = Mock()
        tdwns = KTI(name='Touchdown')
        afr_land_rwy = None
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_rwy, afr_land_apt, lat_c, aeroplane)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, tdwns)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_land_rwy(self):
        lon = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = A(name='AFR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, lon_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_land_apt(self):
        lon = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = None
        afr_land_apt = A(name='AFR Landing Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLongitudeAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeAtLiftoff
        self.operational_combinations = [
            ('Longitude', 'Liftoff'),
            ('Liftoff', 'AFR Takeoff Airport'),
            ('Liftoff', 'AFR Takeoff Runway'),
            ('Liftoff', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Runway'),
            ('Longitude', 'Liftoff', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Liftoff', 'AFR Takeoff Airport', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Longitude')
        lon.array = Mock()
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_rwy, afr_toff_apt, None, aeroplane)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, liftoffs)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_toff_rwy(self):
        lon = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = A(name='AFR Takeoff Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, lon_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_toff_apt(self):
        lon = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = A(name='AFR Takeoff Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLatitudeOffBlocks(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeOffBlocks
        self.operational_combinations = [
            ('Latitude', 'Off Blocks'),
            ('Off Blocks', 'Latitude (Coarse)'),
            ('Latitude', 'Off Blocks', 'Latitude (Coarse)')
        ]

    def test_derive_with_Latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        tdwns = KTI(name='Touchdown')
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, lat_c)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, tdwns)
        assert not node.create_kpv.called, 'method should not have been called'


class TestLongitudeOffBlocks(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeOffBlocks
        self.operational_combinations = [
            ('Longitude', 'Off Blocks'),
            ('Off Blocks', 'Longitude (Coarse)'),
            ('Longitude', 'Off Blocks', 'Longitude (Coarse)')
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Longitude')
        lon.array = Mock()
        liftoffs = KTI(name='Liftoff')
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, None)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, liftoffs)
        assert not node.create_kpv.called, 'method should not have been called'


class TestLatitudeAtTakeoffAccelerationStart(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeAtTakeoffAccelerationStart
        self.operational_combinations = [
            ('Latitude', 'Takeoff Acceleration Start'),
            ('Takeoff Acceleration Start', 'Latitude (Coarse)'),
            ('Latitude', 'Takeoff Acceleration Start', 'Latitude (Coarse)')
        ]

    def test_derive_with_Latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        toff_accel = KTI(name='Takeoff Acceleration Start')
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, toff_accel, lat_c)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, toff_accel)
        assert not node.create_kpv.called, 'method should not have been called'


class TestLongitudeAtTakeoffAccelerationStart(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeAtTakeoffAccelerationStart
        self.operational_combinations = [
            ('Longitude', 'Takeoff Acceleration Start'),
            ('Takeoff Acceleration Start', 'Longitude (Coarse)'),
            ('Longitude', 'Takeoff Acceleration Start', 'Longitude (Coarse)')
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Longitude')
        lon.array = Mock()
        toff_accel = KTI(name='Takeoff Acceleration Start')
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, toff_accel, None)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, toff_accel)
        assert not node.create_kpv.called, 'method should not have been called'


class TestLongitudeSmoothedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeSmoothedAtTouchdown
        self.operational_combinations = [('Longitude Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLongitudeSmoothedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeSmoothedAtLiftoff
        self.operational_combinations = [('Longitude Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLongitudeAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Longitude Prepared', 'Lowest Altitude During Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


################################################################################
# Magnetic Variation


class TestMagneticVariationAtTakeoffTurnOntoRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = MagneticVariationAtTakeoffTurnOntoRunway
        self.operational_combinations = [('Magnetic Variation', 'Takeoff Turn Onto Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariationAtLandingTurnOffRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = MagneticVariationAtLandingTurnOffRunway
        self.operational_combinations = [('Magnetic Variation', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


###############################################################################

class TestIsolationValveOpenAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = IsolationValveOpenAtLiftoff
        self.operational_combinations = [('Isolation Valve Open', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPackValvesOpenAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PackValvesOpenAtLiftoff
        self.operational_combinations = [('Pack Valves Open', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDescentToFlare(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearExtending(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGoAround5MinRating(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLevelFlight(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoff5MinRating(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRoll(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRotation(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestElevatorDuringLandingMin(unittest.TestCase,
                                   CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ElevatorDuringLandingMin
        self.operational_combinations = [('Elevator', 'Landing')]
        self.function = min_value

    def test_derive(self):
        ccf = P(
            name='Elevator During Landing',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Landing', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Elevator During Landing Min',
                items=[KeyPointValue(
                    index=9.0, value=41.0,
                    name='Elevator During Landing Min')]))


class TestHeadingVariationAbove80KtsAirspeedDuringTakeoff(unittest.TestCase, NodeTest):
    nosewheel=P('Gear (N) On Ground',array=np.ma.array([1]*9+[0]*2))
    hdg = P('Heading True Continuous', np.ma.array([45]*11))
    ias = P('Airspeed', np.ma.arange(60, 170, 10))
    q = P('Pitch Rate', np.ma.array([0]*8+[1, 2, 3]))
    toff = buildsection('Takeoff', 1, 10)
    '''
    KPV used to use runway details. Retained for possible future re-use.
    rwy = A(name='FDR Takeoff Runway', value={
        'start': {'latitude': -.01, 'longitude': -.01},
        'end': {'latitude': +0.01, 'longitude': +0.01},
        'identifier':'040',})
    '''

    def setUp(self):
        self.node_class = HeadingVariationAbove80KtsAirspeedDuringTakeoff
        self.operational_combinations = [(
            'Heading True Continuous',
            'Airspeed',
            'Pitch Rate',
            'Takeoff',
        )]
        self.can_operate_kwargs = {'ac_type': aeroplane}


    def test_no_deviation(self):
        node = self.node_class()
        node.derive(self.nosewheel, self.hdg, None, self.ias, self.q, self.toff)
        self.assertAlmostEqual(node[0].value, 0.0, places=5)

    def test_with_deviation(self):
        self.hdg.array[8] = 40.0
        node = self.node_class()
        node.derive(self.nosewheel, self.hdg, None, self.ias, self.q, self.toff)
        self.assertAlmostEqual(node[0].value, -5.0, places=5)

    def test_with_transient_deviation_at_1_5_deg_sec(self):
        self.hdg = P('Heading True Continuous', np.ma.array([45]*8+[50.0, 55, 60]))
        # Pitch rate passes 1.5 deg as heading goes through 52.5 = +7.5 deg from datum.
        self.pch_rate = P('Pitch', np.ma.array([0]*8+[1.0, 2, 2]))
        node = self.node_class()
        node.derive(None, self.hdg, None, self.ias, self.q, self.toff)
        self.assertAlmostEqual(node[0].value, 7.5, places=5)

    def test_with_transient_deviation_at_80_kts(self):
        self.hdg = P('Heading True Continuous', np.ma.arange(35,46))
        self.ias = P('Airspeed', np.ma.arange(55, 165, 10))
        node = self.node_class()
        node.derive(None, self.hdg, None, self.ias, self.q, self.toff)
        self.assertAlmostEqual(node[0].value, 7.0, places=5)
        self.assertEqual(node[0].index, 8.5)

    def test_nosewheel_didnt_lift(self):
        self.nosewheel.array[8:] = [1]*3
        node = self.node_class()
        node.derive(self.nosewheel, self.hdg, None, self.ias, self.q, self.toff)
        self.assertEqual(node[0].index, 2.0)

    def test_derive__stationary_off_centre(self):
        '''
        Test for when aircraft stops off centre at start of runway,
        streightens as starting to move down runway. should be no variation
        above 80kts.
        '''
        hdg = P('Heading True Continuous', np.ma.array([55]*10 + [65]*8))
        ias = P('Airspeed', np.ma.array([0]*12 + list(range(60, 120, 10))))
        ias.array = np.ma.masked_less(ias.array, 10)
        q = P('Pitch Rate', np.ma.array([0]*15+[1, 2, 3]))
        toff = buildsection('Takeoff', 1, 18)
        node = self.node_class()
        node.derive(None, hdg, None, ias, q, toff)
        self.assertAlmostEqual(node[0].value, 0.0, places=5)

    def test_derive_heading_masked(self):

        def load_param(name):
            array = load_compressed(os.path.join(
                test_data_path, 'HeadingVariationAbove80KtsAirspeedDuringTakeoff_%s.npz' % name))
            cls = M if isinstance(array, MappedArray) else P
            return cls(name, array, frequency=0.25)


        nosewheel = load_param('Gear')
        head_true = load_param('HeadingTrueContinuous')
        head_mag = load_param('HeadingContinuous')
        airspeed = load_param('Airspeed')
        pitch_rate = load_param('PitchRate')

        toffs = buildsection('Takeoff', 96, 108)
        toffs.frequency = 0.25

        node = self.node_class()
        node.derive(nosewheel, head_true, head_mag, airspeed, pitch_rate, toffs)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 107, places=2)
        self.assertAlmostEqual(node[0].value, -3.29, places=2)


class TestHeadingDeviationFromRunwayAtTOGADuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayAtTOGADuringTakeoff
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Takeoff And Go Around',
            'Heading True Continuous',
            'Takeoff',
            'FDR Takeoff Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayAt50FtDuringLanding(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayAt50FtDuringLanding
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Heading True Continuous',
            'Landing',
            'FDR Landing Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayDuringLandingRoll(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayDuringLandingRoll
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Heading True Continuous',
            'Landing Roll',
            'FDR Landing Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariation300To50Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = HeadingVariation300To50Ft

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate(['Heading Continuous', 'Altitude AAL For Flight Phases'], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(['Heading Continuous', 'Altitude AAL For Flight Phases'], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(['Heading Continuous', 'Altitude AGL', 'Descending'], ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariation500To50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariation500To50Ft
        self.operational_combinations = [(
            'Heading Continuous',
            'Altitude AAL For Flight Phases',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

class TestHeadingVariation800To50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariation800To50Ft
        self.operational_combinations = [(
            'Heading Continuous',
            'Altitude AAL For Flight Phases',
        )]

    def test_derive(self):
        hdg = P('Heading Continuous',             array=[100, 100, 101, 102, 103, 102, 101, 100, 99, 98, 98, 96])
        alt = P('Altitude AAL For Flight Phases', array=[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25])
        node = self.node_class()
        node.derive(hdg, alt)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 5.0)

        #self.assertTrue(False, msg='Test not implemented.')

class TestHeadingVariationAbove100KtsAirspeedDuringLanding(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariationAbove100KtsAirspeedDuringLanding
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [(
            'Heading Continuous',
            'Airspeed',
            'Altitude AAL For Flight Phases',
            'Landing',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariationTouchdownPlus4SecTo60KtsAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariationTouchdownPlus4SecTo60KtsAirspeed
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Heading Continuous', 'Airspeed True', 'Touchdown')]

    def test_derive(self):

        heading = P('Heading Continuous', np.ma.array([10]*25))
        heading.array[15] = 15
        # This value ensures the array is not artificially "quiet":
        heading.array[18] = 11
        heading.array[-5] = 20
        airspeed = P('Airspeed True', np.ma.arange(99, 50, -2))
        airspeed.array[-5:] = np.ma.masked
        tdwns = KTI(name='Touchdown', items=[
            KeyTimeInstance(index=10, name='Touchdown'),
        ])
        node = self.node_class()
        node.derive(heading, airspeed, tdwns)
        self.assertEqual(len(node), 1, msg="Expected one KPV got %s" % len(node))
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[0].index, 15)

    def test_ignore_ret(self):
        heading = P('Heading Continuous', np.ma.array([10]*25))
        heading.array[10] = 5
        heading.array[15] = 15
        # The final samples increase to represent a rapid exit turnoff.
        heading.array[-4:] = [12, 18, 24, 30]
        airspeed = P('Airspeed True', np.ma.arange(107, 58, -2))
        tdwns = KTI(name='Touchdown', items=[
            KeyTimeInstance(index=4, name='Touchdown'),
        ])
        hvt = HeadingVariationTouchdownPlus4SecTo60KtsAirspeed()
        hvt.derive(heading, airspeed, tdwns)
        self.assertEqual(hvt[0].value, 10)
        self.assertEqual(hvt[0].index, 10)

    def test_derive_straight(self):

        heading = P('Heading Continuous', np.ma.array([253.8]*35))
        spd_array = np.ma.array([np.ma.masked]*35)
        spd_array[:26] = np.ma.arange(118.8, 61.6, -2.2)
        airspeed = P('Airspeed True', spd_array)
        tdwns = KTI(name='Touchdown', items=[
            KeyTimeInstance(index=6, name='Touchdown'),
        ])
        node = self.node_class()
        node.derive(heading, airspeed, tdwns)
        self.assertEqual(len(node), 0, msg="Expected zero KPVs got %s" % len(node))

    def test_first_sample_interpolation(self):
        heading = P('Heading Continuous', np.ma.array([0]*5+[1, 1, 2, 1, 1.0]))
        airspeed = P('Airspeed True', np.ma.array([100]*9+[0]))
        tdwns = KTI(name='Touchdown', items=[
            KeyTimeInstance(index=0.5, name='Touchdown'),])
        hvt = HeadingVariationTouchdownPlus4SecTo60KtsAirspeed()
        hvt.derive(heading, airspeed, tdwns)
        self.assertEqual(hvt[0].value, 1.5)
        self.assertEqual(hvt[0].index, 4.5)




class TestHeadingVacatingRunway(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVacatingRunway
        self.operational_combinations = [('Heading Continuous', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Height


class TestHeightMinsToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeightMinsToTouchdown
        self.operational_combinations = [('Altitude AAL', 'Mins To Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Flap


class TestFlapAtFirstMovementAfterEngineStart(unittest.TestCase):
    def setUp(self):
        self.node_class = FlapAtFirstMovementAfterEngineStart

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Flap At First Movement After Engine Start')
        self.assertEqual(node.units, ut.DEGREE)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertIn('Flap Including Transition', opts[0])
        self.assertIn('Groundspeed Signed', opts[0])
        self.assertIn('First Eng Start Before Liftoff', opts[0])
        self.assertIn('Liftoff', opts[0])

    def test_derive(self):

        flap = P('Flap', np.ma.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 15, 15, 15, 15,
            15, 15, 15, 15, 0
        ]))
        gnd_spd = P('Groundspeed Signed', np.ma.array([
            0., 5., 0., 0., 0.,
            0., 0., -1., -3., 0.,
            0., 0., 1., 2., 5.,
            10., 15., 20., 25., 30.
        ]))
        eng_start = KTI('First Eng Start Before Liftoff', items=[
            KeyTimeInstance(3,'First Eng Start Before Liftoff'),
        ])
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(19, 'Liftoff')])

        node = self.node_class()
        node.derive(flap=flap, gnd_spd=gnd_spd, eng_start=eng_start,
                    liftoff=liftoff)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 15)

    def test_derive_tow(self):
        flap = P('Flap', np.ma.array([
            0, 0, 0, 0, 0, # before eng start
            0, 0, 0, 0, 0,
            1, 15, 15, 15, 15,
            15, 15, 15, 15, 0
        ]))
        gnd_spd = P('Groundspeed Signed', np.ma.array([
            0., 5., 0., 0., 0., # before eng start
            0., 3., 4., 0., 0., # towed
            0., 0., 1., 2., 5.,
            10., 15., 20., 25., 30.
        ]))
        eng_start = KTI('First Eng Start Before Liftoff', items=[
            KeyTimeInstance(3,'First Eng Start Before Liftoff'),
        ])
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(19, 'Liftoff')])

        node = self.node_class()
        node.derive(flap=flap, gnd_spd=gnd_spd, eng_start=eng_start,
                    liftoff=liftoff)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 15)


class TestFlapAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAtLiftoff
        self.operational_combinations = [
            ('Flap', 'Liftoff'),
        ]

    def test_derive(self):
        array = np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        for index, value in (14.25, 5), (14.75, 15), (15.00, 15), (15.25, 15):
            liftoffs = KTI(name='Liftoff', items=[
                KeyTimeInstance(index=index, name='Liftoff'),
            ])
            name = self.node_class.get_name()
            node = self.node_class()
            node.derive(flap, liftoffs)
            self.assertEqual(node, KPV(name=name, items=[
                KeyPointValue(index=index, value=value, name=name),
            ]))

    def test_derive__lfl_multistate(self):
        '''
        Test for lfl multistates which may be pulled together from multiple
        discreats. Values will not be equall to states.
        '''
        flap_mapping = {8: '39', 1: '0', 2: '10', 4: '20'}
        array = np.ma.repeat((1, 2, 4, 8), 10)
        array.mask = np.ma.getmaskarray(array)
        flap_array = MappedArray(array, values_mapping=flap_mapping)
        flap = M(name='Flap', array=flap_array)
        for index, value in (5, 0), (15, 10), (25, 20), (35, 39):
            liftoffs = KTI(name='Liftoff', items=[
                KeyTimeInstance(index=index, name='Liftoff'),
            ])
            name = self.node_class.get_name()
            node = self.node_class()
            node.derive(flap, liftoffs)
            self.assertEqual(node, KPV(name=name, items=[
                KeyPointValue(index=index, value=value, name=name),
            ]))


class TestFlapAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAtTouchdown
        self.operational_combinations = [
            ('Flap', 'Touchdown'),
        ]

    def test_derive(self):
        array = np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        for index, value in (29.25, 25), (29.75, 30), (30.00, 30), (30.25, 30):
            touchdowns = KTI(name='Touchdown', items=[
                KeyTimeInstance(index=index, name='Touchdown'),
            ])
            name = self.node_class.get_name()
            node = self.node_class()
            node.derive(flap, touchdowns)
            self.assertEqual(node, KPV(name=name, items=[
                KeyPointValue(index=index, value=value, name=name),
            ]))

    def test_derive__lfl_multistate(self):
        '''
        Test for lfl multistates which may be pulled together from multiple
        discreats. Values will not be equall to states.
        '''
        flap_mapping = {8: '39', 1: '0', 2: '10', 4: '20'}
        array = np.ma.repeat((1, 2, 4, 8), 10)
        array.mask = np.ma.getmaskarray(array)
        flap_array = MappedArray(array, values_mapping=flap_mapping)
        flap = M(name='Flap', array=flap_array)
        for index, value in (5, 0), (15, 10), (25, 20), (35, 39):
            liftoffs = KTI(name='Touchdown', items=[
                KeyTimeInstance(index=index, name='Touchdown'),
            ])
            name = self.node_class.get_name()
            node = self.node_class()
            node.derive(flap, liftoffs)
            self.assertEqual(node, KPV(name=name, items=[
                KeyPointValue(index=index, value=value, name=name),
            ]))


class TestFlapAtGearDownSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAtGearDownSelection
        self.operational_combinations = [
            ('Flap', 'Gear Down Selection'),
        ]

    def test_derive(self):
        array = np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        flap.array[29] = np.ma.masked
        gear = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=19.25, name='Gear Down Selection'),
            KeyTimeInstance(index=19.75, name='Gear Down Selection'),
            KeyTimeInstance(index=20.00, name='Gear Down Selection'),
            KeyTimeInstance(index=20.25, name='Gear Down Selection'),
            KeyTimeInstance(index=29.25, name='Gear Down Selection'),
            KeyTimeInstance(index=29.75, name='Gear Down Selection'),
            KeyTimeInstance(index=30.00, name='Gear Down Selection'),
            KeyTimeInstance(index=30.25, name='Gear Down Selection'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap, gear)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=19.25, value=15, name=name),
            KeyPointValue(index=19.75, value=20, name=name),
            KeyPointValue(index=20.00, value=20, name=name),
            KeyPointValue(index=20.25, value=20, name=name),
            # Note: Index 29 is masked so we get a value of 30, not 25!
            KeyPointValue(index=29.25, value=30, name=name),
            KeyPointValue(index=29.75, value=30, name=name),
            KeyPointValue(index=30.00, value=30, name=name),
            KeyPointValue(index=30.25, value=30, name=name),
        ]))


class TestFlapAtGearUpSelectionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAtGearUpSelectionDuringGoAround
        self.operational_combinations = [('Flap', 'Gear Up Selection During Go Around')]

    def test_derive(self):
        array = np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        flap.array[29] = np.ma.masked
        gear = KTI(name='Gear Up Selection During Go Around', items=[
            KeyTimeInstance(index=19.25, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=19.75, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=20.00, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=20.25, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=29.25, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=29.75, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=30.00, name='Gear Up Selection During Go Around'),
            KeyTimeInstance(index=30.25, name='Gear Up Selection During Go Around'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap, gear)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=19.25, value=15, name=name),
            KeyPointValue(index=19.75, value=20, name=name),
            KeyPointValue(index=20.00, value=20, name=name),
            KeyPointValue(index=20.25, value=20, name=name),
            # Note: Index 29 is masked so we get a value of 30, not 25!
            KeyPointValue(index=29.25, value=30, name=name),
            KeyPointValue(index=29.75, value=30, name=name),
            KeyPointValue(index=30.00, value=30, name=name),
            KeyPointValue(index=30.25, value=30, name=name),
        ]))


class TestFlapLeverAtGearDownSelection(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = FlapLeverAtGearDownSelection
        self.operational_combinations = [('Flap Lever',),('Gear Down Selection',)]
        self.values_mapping = {
            1:  'Level 0',
            2:  'Level 1',
            4:  'Level 2',
            8:  'Level 3',
            16: 'Level 4',
            32: 'Level 5',
            64: 'Level 6',
        }

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),
                                                    family=A('Family','ERJ-170/175')))
        self.assertTrue(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),
                                                    family=A('Family','ERJ-170/175')))
        self.assertFalse(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),
                                                     family=A('Family','Learjet')))

    def test_derive(self):
        array = np.ma.repeat((4,1,1,4,32,1), 5)
        flap_lever = M(name='Flap Lever', array=array, values_mapping=self.values_mapping)
        gear = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=21, name='Gear Down Selection',)
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap_lever, gear)

        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=21, value=5, name=name),
        ]))


class TestFlapLeverAtGearUpSelectionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLeverAtGearUpSelectionDuringGoAround
        self.operational_combinations = [('Flap Lever', 'Gear Up Selection During Go Around')]
        self.values_mapping = {
            1:  'Level 0',
            2:  'Level 1',
            4:  'Level 2',
            8:  'Level 3',
            16: 'Level 4',
            32: 'Level 5',
            64: 'Level 6',
        }

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),family=A('Family','ERJ-170/175')))
        self.assertTrue(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),family=A('Family','ERJ-170/175')))
        self.assertFalse(self.node_class.can_operate(('Flap Lever','Gear Down Selection',),family=A('Family','Learjet')))

    def test_derive(self):
        array = np.ma.repeat((4,1,1,4,32,1,2,4,32), 5)

        flap_lever = M(name='Flap Lever', array=array, values_mapping=self.values_mapping)

        gear = KTI(name='Gear Up Selection During Go Around', items=[
            KeyTimeInstance(index=25, name='Gear Up Selection During Go Around'),
        ])

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap_lever, gear)

        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=25, value=np.log2(array[25]), name=name),
        ]))



class TestFlapOrConfigurationMaxOrMin(unittest.TestCase):

    def test_flap_or_conf_max_or_min_empty_scope(self):
        flap = M('Flap', array=np.ma.zeros(10), values_mapping={0: '0'})
        result = FlapOrConfigurationMaxOrMin.flap_or_conf_max_or_min(
            flap, np.arange(10), max_value, [])
        self.assertEqual(result, [])


class TestFlapWithGearUpMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapWithGearUpMax
        self.operational_combinations = [
            ('Flap', 'Gear Down'),
        ]

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestFlapWithSpeedbrakeDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapWithSpeedbrakeDeployedMax
        self.operational_combinations = [
            ('Flap Including Transition', 'Speedbrake Selected', 'Airborne', 'Landing',
             'Landing Attitude Mode Delta CL',),
        ]

    def test_derive(self):
        array = np.ma.repeat((0, 7, 15), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        spd_brk = M(
            name='Speedbrake Selected',
            array=np.ma.array([0, 1, 2, 0, 0] * 3),
            values_mapping={
                0: 'Stowed',
                1: 'Armed/Cmd Dn',
                2: 'Deployed/Cmd Up',
                },
        )
        airborne = buildsection('Airborne', 5, 15)
        landings = buildsection('Landing', 10, 15)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap, spd_brk, airborne, landings, None)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7, value=7, name=name),
        ]))

    def test_737MAX(self):
        array = np.ma.repeat((0, 7, 15), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        spd_brk = M(
            name='Speedbrake Selected',
            array=np.ma.array([0, 1, 2, 2, 0] * 3),
            values_mapping={
                0: 'Stowed',
                1: 'Armed/Cmd Dn',
                2: 'Deployed/Cmd Up',
                },
        )
        airborne = buildsection('Airborne', 5, 15)
        landings = buildsection('Landing', 10, 15)
        lam = P(
            name = 'Landing Attitude Mode Delta CL',
                array = [0, 0, 10, 0, 0]*3)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap, spd_brk, airborne, landings, lam)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=8, value=7, name=name),
        ]))


class TestFlapAt1000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAt1000Ft
        self.operational_combinations = [
            ('Flap', 'Altitude When Descending'),
        ]

    def test_derive(self):
        array = np.ma.concatenate((np.ones(70) * 30, np.ones(30) * 45))
        mapping = {30: '30', 45: '45'}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        gates = AltitudeWhenDescending(items=[
            KeyTimeInstance(index=40, name='1500 Ft Descending'),
             KeyTimeInstance(index=60, name='1000 Ft Descending'),
             KeyTimeInstance(index=80, name='1000 Ft Descending'),  # 2nd descent after G/A?
             KeyTimeInstance(index=90, name='500 Ft Descending'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap=flap, gates=gates)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=60, value=30, name=name),
            KeyPointValue(index=80, value=45, name=name),
        ]))


class TestFlapAt500Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAt500Ft
        self.operational_combinations = [
            ('Flap', 'Altitude When Descending'),
        ]

    def test_derive(self):
        array = np.ma.concatenate((np.ones(70) * 30, np.ones(30) * 45))
        mapping = {30: '30', 45: '45'}
        flap = M(name='Flap', array=array, values_mapping=mapping)
        gates = AltitudeWhenDescending(items=[
            KeyTimeInstance(index=40, name='1500 Ft Descending'),
             KeyTimeInstance(index=60, name='1000 Ft Descending'),
             KeyTimeInstance(index=80, name='1000 Ft Descending'),  # 2nd descent after G/A?
             KeyTimeInstance(index=90, name='500 Ft Descending'),
        ])
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(flap=flap, gates=gates)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=90, value=45, name=name),
        ]))


class TestFlapSynchroAsymmetryMax(unittest.TestCase):
    def test_derive(self):
        synchro = P('Flap Synchro Asymmetry', [0,0,1,4,5,6,6,6,9,8,3,0,0,0])
        asym = FlapSynchroAsymmetryMax()
        asym.derive(synchro)
        self.assertEqual(asym[0].index, 8)
        self.assertEqual(asym[0].value, 9)


class TestFlapBypassValveDuration(unittest.TestCase):
    def test_derive(self):
        valve = M('Flap Bypass Valve Position',
                  array=np.ma.array([0,0,1,1,1,0,0,1,1,1,1,1,0,0]),
                  values_mapping={0: '-', 1: 'Bypass'})
        node=FlapBypassValveDuration()
        node.derive(valve)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[1].index, 7)
        self.assertEqual(node[1].value, 5)


class TestGearDownToLandingFlapConfigurationDuration(unittest.TestCase):

    class ERJ(VelocitySpeed):
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': (30,  40),
            'Lever 4': (110, 120),
            'Lever 5': (110, 120),
            'Lever Full': (120, 130),
        }}

    class Phenom(VelocitySpeed):
        weight_unit = ut.TONNE
        tables = {'vapp': {
            'weight': (30,  40),
            'Lever 3': (110, 120),
            'Lever Full': (120, 130),
        }}

    ATTRS = {
        'family': A('Family', 'B737'),
        'series': A('Series', 'B737-300'),
        'model': A('Model', 'B737-3Q8'),
        'engine_type': A('Engine Type', 'CFM56-3B1'),
        'engine_series': A('Engine Series', 'CFM56-3'),
    }

    def setUp(self):
        self.node_class = GearDownToLandingFlapConfigurationDuration
        self.values_mapping = {
            0: 'Lever 0',
            10: 'Lever 1',
            20: 'Lever 2',
            30: 'Lever 3',
            40: 'Lever 4',
            50: 'Lever 5',
            90: 'Lever Full',
        }
        self.reverse_lookup = {v: k for k, v in self.values_mapping.items()}

    @patch('analysis_engine.key_point_values.lookup_table')
    def test_can_operate(self, lookup_table):

        lookup_table.return_value = self.ERJ

        self.assertFalse(self.node_class.can_operate([], **self.ATTRS))
        self.assertFalse(self.node_class.can_operate(['Gear Down Selection', 'Approach And Landing'], **self.ATTRS))
        self.assertTrue(self.node_class.can_operate(['Flap Lever', 'Gear Down Selection', 'Approach And Landing'], **self.ATTRS))
        self.assertTrue(self.node_class.can_operate(['Flap Lever (Synthetic)', 'Gear Down Selection', 'Approach And Landing'], **self.ATTRS))
        self.assertTrue(self.node_class.can_operate(['Flap Lever', 'Flap Lever (Synthetic)', 'Gear Down Selection', 'Approach And Landing'], **self.ATTRS))

    @patch('analysis_engine.key_point_values.lookup_table')
    def test_derive_basic_phenom_300(self, lookup_table):
        lookup_table.return_value = self.Phenom()

        flap_lever_values = np.ma.array(
            [self.reverse_lookup['Lever 0']] * 10 +
            [self.reverse_lookup['Lever 1']] * 10 +
            [self.reverse_lookup['Lever 2']] * 10 +
            [self.reverse_lookup['Lever 3']] * 10 +
            [self.reverse_lookup['Lever Full']] * 10
        )

        flap_lever = M('Flap Lever', array=flap_lever_values, values_mapping=self.values_mapping)

        gear_dn_sel = KTI('Gear Down Selection',
                          items=[KeyTimeInstance(x) for x in (5, 15, 25, 27, 41)])

        approaches = buildsections('Approach And Landing', (15, 20), (20, 32), (35, 45))

        def test_assertions(node):
            self.assertEqual(len(node), 2)
            self.assertEqual(node[0].index, 29.5)
            self.assertEqual(node[0].value, 2.5)
            self.assertEqual(node[1].index, 39.5)
            self.assertEqual(node[1].value, -1.5)

        node = self.node_class()
        node.derive(flap_lever, None, gear_dn_sel, approaches, **self.ATTRS)
        test_assertions(node)

        node = self.node_class()
        node.derive(None, flap_lever, gear_dn_sel, approaches, **self.ATTRS)
        test_assertions(node)

        node = self.node_class()
        node.derive(flap_lever, flap_lever, gear_dn_sel, approaches, **self.ATTRS)
        test_assertions(node)

    @patch('analysis_engine.key_point_values.lookup_table')
    def test_derive_basic_erj(self, lookup_table):
        lookup_table.return_value = self.ERJ()

        flap_lever_values = np.ma.array(
            [self.reverse_lookup['Lever 0']] * 10 +
            [self.reverse_lookup['Lever 1']] * 10 +
            [self.reverse_lookup['Lever 2']] * 10 +
            [self.reverse_lookup['Lever 3']] * 10 +
            [self.reverse_lookup['Lever 4']] * 10 +
            [self.reverse_lookup['Lever 5']] * 10 +
            [self.reverse_lookup['Lever Full']] * 10
        )

        flap_lever = M('Flap Lever', array=flap_lever_values, values_mapping=self.values_mapping)

        gear_dn_sel = KTI('Gear Down Selection',
                          items=[KeyTimeInstance(x) for x in (5, 15, 25, 27, 41, 52)])

        approaches = buildsections('Approach And Landing', (15, 20), (20, 32), (35, 45), (45, 55), (55, 65))

        node = self.node_class()
        node.derive(flap_lever, None, gear_dn_sel, approaches, **self.ATTRS)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 39.5)
        self.assertEqual(node[0].value, -1.5)
        self.assertEqual(node[1].index, 49.5)
        self.assertEqual(node[1].value, -2.5)


##############################################################################


class TestFlareDuration20FtToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = FlareDuration20FtToTouchdown
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Touchdown', 'Landing', 'Altitude Radio')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlareDistance20FtToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = FlareDistance20FtToTouchdown
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [
            ('Altitude AAL For Flight Phases',
             'Touchdown', 'Landing', 'Groundspeed')]

    def test_derive(self):
        flare_dist = FlareDistance20FtToTouchdown()
        alt_aal = P('Altitude AAL', frequency=2, array=np.ma.array([
            78.67015114,  72.96263885,  67.11249936,  61.07833929,
            54.90942938,  48.68353449,  42.50324374,  36.48330464,
            30.76228438,  25.47636721,  20.7296974 ,  16.6029949 ,
            13.11602541,  10.26427979,   8.00656174,   6.28504293,
            5.0119151 ,   4.09266636,   3.43519846,   2.95430818,
            2.58943048,   2.29123077,   2.03074567,   1.78243098,
            1.51466402,   1.19706783,   0.80993319,   0.3440701 ,
            0.        ,   0.        ,   0.        ,   0.        ,
            0.        ,   0.        ,   0.        ,   0.        ,
            0.        ,   0.        ]))

        gspd = P('Groundspeed', frequency=1, array=np.ma.array([
            144.40820312,  144.5       ,  144.5       ,  144.5       ,
            144.5       ,  144.5       ,  144.5       ,  144.65820312,
            144.90820312,  145.        ,  145.        ,  144.84179688,
            144.59179688,  144.18359375,  143.68359375,  143.18359375,
            142.68359375,  142.02539062,  141.27539062,  140.52539062,
            139.77539062,  139.02539062,  138.27539062,  137.52539062,
            136.77539062,  136.02539062,  135.27539062,  134.52539062,
            133.77539062,  132.8671875 ,  131.8671875 ,  130.8671875 ,
            129.8671875 ,  128.70898438,  127.45898438,  125.89257812,
            124.14257812,  122.39257812]))
        tdwn = KTI('Touchdown', frequency=2)
        tdwn.create_kti(27.0078125)
        ldg = S('Landing', frequency=2)
        ldg.create_section(slice(5, 28, None),
                           begin=5.265625, end=27.265625)
        flare_dist.get_derived([alt_aal, tdwn, ldg, gspd])
        self.assertAlmostEqual(flare_dist[0].index, 27, 0)
        self.assertAlmostEqual(flare_dist[0].value, 632.69, 0)  # Meters


##############################################################################
# Fuel Quantity


class TestFuelQtyAtLiftoff(unittest.TestCase):

    def test_can_operate(self):
        opts = FuelQtyAtLiftoff.get_operational_combinations()
        self.assertEqual(opts, [('Fuel Qty', 'Liftoff')])

    def test_derive(self):
        # example from B777 recorded fuel qty parameter
        fuel_qty = P('Fuel Qty', np.ma.array(
            [ 105600.,  105600.,  105600.,  105600.,  105600.,  105500.,
              105500.,  105500.,  105500.,  105500.,  105500.,  105500.,
              105500.,  105500.,  105500.,  105500.,  105500.,  105500.,
              105500.,  105600.,  105600.,  105600.,  105500.,  105500.,
              105500.,  105500.,  105500.,  105500.,  105500.,  105500.,
              105500.,  105400.,  105400.,  105500.,  105500.,  105500.,
              105400.,  105400.,  105400.,  105400.,  105400.,  105400.,
              105400.,  105400.,  105400.,  105500.,  105500.,  105600.,
              105500.,  105500.,  105400.,  105400.,  105300.,  105100.,
              105300.,  105300.,  105300.,  105300.,  105400.,  105400.,
              105300.,  105300.,  105200.,  105300.,  105400.,  105400.,
              105400.,  105500.,  105600.,  105500.,  105500.,  105500.,
              105500.,  105500.,  105400.,  105500.,  105500.,  105400.,
              105400.,  105400.,  105500.,  105500.,  105400.,  105400.,
              105500.,  105400.,  105400.,  105400.,  105300.,  105300.,
              105300.,  105200.,  105200.,  105200.,  105100.,  105100.,
              105100.,  105100.,  104900.,  104900.]))
        # roc limit exceeded, caused by long G at liftoff
        fuel_qty.array[53] = np.ma.masked
        fuel_qty.array[54] = np.ma.masked
        fuel_qty.array[58] = np.ma.masked
        liftoff = KTI(items=[KeyTimeInstance(54)])
        fq = FuelQtyAtLiftoff()
        fq.derive(fuel_qty, liftoff)
        self.assertEqual(len(fq), 1)
        self.assertEqual(fq[0].index, 54)
        self.assertAlmostEqual(fq[0].value, 105371, 0)


class TestFuelQtyAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQtyAtTouchdown.get_operational_combinations()
        self.assertEqual(opts, [('Fuel Qty', 'Touchdown')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestFuelQtyWingDifferenceMax(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQtyWingDifferenceMax.get_operational_combinations()
        self.assertEqual(opts, [('Fuel Qty (L)', 'Fuel Qty (R)', 'Airborne')])

    def test_derive_basic(self):
        qty_l = P('Fuel Qty (L)', array=np.ma.array([100, 90, 80, 70, 60, 50]))
        qty_r = P('Fuel Qty (R)', array=np.ma.array([110, 100, 95, 80, 70, 60]))
        airs = buildsection('Airborne', 1, 4)
        node = FuelQtyWingDifferenceMax()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 15)

    def test_derive_handed(self):
        qty_r = P('Fuel Qty (R)', array=np.ma.array([100, 90, 80, 70, 60, 50]))
        qty_l = P('Fuel Qty (L)', array=np.ma.array([110, 100, 95, 80, 70, 60]))
        airs = buildsection('Airborne', 1, 4)
        node = FuelQtyWingDifferenceMax()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, -15)

class TestFuelQtyWingDifference787Max(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQtyWingDifference787Max.get_operational_combinations(frame=A('Frame', value='787_frame'))
        self.assertEqual(opts, [('Fuel Qty (L)', 'Fuel Qty (R)', 'Airborne')])

    def test_derive_basic(self):
        '''
        These values are on the permitted imbalance line, below, on, midway and above the turning points.
        '''
        qty_l = P('Fuel Qty (L)', array=np.ma.array([16150, 18100, 27050, 32400, 35650]), frequency=1.0/16)
        qty_r = P('Fuel Qty (R)', array=np.ma.array([13850, 20400, 25250, 33700, 34350]), frequency=1.0/16)
        airs = buildsection('Airborne', 0, 6)
        node = FuelQtyWingDifference787Max()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, -100.0)

    def test_derive_handed(self):
        qty_l = P('Fuel Qty (L)', array=np.ma.array([16150, 20400, 27050, 33700, 35650]), frequency=1.0/16)
        qty_r = P('Fuel Qty (R)', array=np.ma.array([13850, 18100, 25250, 32400, 34350]), frequency=1.0/16)
        airs = buildsection('Airborne', 0, 6)
        node = FuelQtyWingDifference787Max()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, -100.0)
        qty_r = P('Fuel Qty (R)', array=np.ma.array([16150, 20400, 27050, 33700, 35650]), frequency=1.0/16)
        qty_l = P('Fuel Qty (L)', array=np.ma.array([13850, 18100, 25250, 32400, 34350]), frequency=1.0/16)
        node = FuelQtyWingDifference787Max()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 100.0)

    def test_masked_operation(self):
        qty_l = P('Fuel Qty (L)', array=np.ma.array(data=[16150, 18100, 27050, 32400, 35650],
                                                    mask=[0,0,1,0,0]), frequency=1.0/16)
        qty_r = P('Fuel Qty (R)', array=np.ma.array([13850, 20400, 25250, 33700, 34350]), frequency=1.0/16)
        airs = buildsection('Airborne', 0, 6)
        node = FuelQtyWingDifference787Max()
        node.derive(qty_l, qty_r, airs)
        self.assertEqual(len(node), 0)


class TestFuelJettisonDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Fuel Jettison Nozzle'
        self.phase_name = 'Airborne'
        self.node_class = FuelJettisonDuration
        self.values_mapping = {0: '-', 1: 'Disagree'}

        self.basic_setup()

class TestFuelCrossFeedValveStateAtLiftoff(unittest.TestCase):

    def setUp(self):
        self.node_class = FuelCrossFeedValveStateAtLiftoff

        self.valve_data=M('Fuel Cross Feed Valve Position',
             array=np.ma.array([0]*5 + [1]*5),
             values_mapping = {0: 'Closed', 1: 'Open'})

        self.valve_data_fallback=M('Fuel Cross Feed Valve',
             array=np.ma.array([0]*5 + [1]*5),
             values_mapping = {0: 'Normal', 1: 'Disagree'})

        self.liftoff_closed = KTI('Liftoff',
                              items=[KeyTimeInstance(4, 'Liftoff')])

        self.liftoff_open = KTI('Liftoff',
                            items=[KeyTimeInstance(6, 'Liftoff')])

    def test_can_operate(self):
        expected = [('Fuel Cross Feed Valve Position',),
                    ('Fuel Cross Feed Valve',)]
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(any_of(expected, opts))

    def test_derive_closed(self):
        node = FuelCrossFeedValveStateAtLiftoff()
        node.derive(self.valve_data, None, self.liftoff_closed)

        self.assertEqual(len(node), 0)

    def test_derive_open(self):
        node = FuelCrossFeedValveStateAtLiftoff()
        node.derive(self.valve_data, None, self.liftoff_open)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 1)

    def test_derive_closed_fallback(self):
        node = FuelCrossFeedValveStateAtLiftoff()
        node.derive(None, self.valve_data_fallback, self.liftoff_closed)

        self.assertEqual(len(node), 0)

    def test_derive_open_fallback(self):
        node = FuelCrossFeedValveStateAtLiftoff()
        node.derive(None, self.valve_data_fallback, self.liftoff_open)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 1)


class TestMainFuelQuantityOffBlockWithFuelInCenterTank(unittest.TestCase):
    def setUp(self):
        self.qty_l = P('Fuel Qty (L)', array=np.ma.array(data=[16150, 18100, 27050, 32400, 35650],
                                                        mask=[0,0,1,0,0]), frequency=1.0/4)
        self.qty_r = P('Fuel Qty (R)', array=np.ma.array([13850, 20400, 25250, 33700, 34350]), frequency=1.0/4)
        self.qty_c = P('Fuel Qty (C)', array=np.ma.array([50, 60, 70, 80, 90]), frequency=1.0/4)

        self.offblocks = KTI('Off Blocks', items=[KeyTimeInstance(3, 'Off Blocks')])

    def test_can_operate(self):
        opts = MainFuelQuantityOffBlockWithFuelInCenterTank.get_operational_combinations()
        self.assertEqual(opts,
                         [('Fuel Qty (L)', 'Fuel Qty (R)',
                           'Fuel Qty (C)', 'Off Blocks')])

    def test_no_ctr_tank_fuel(self):
        node = MainFuelQuantityOffBlockWithFuelInCenterTank()
        node.derive(self.qty_l, self.qty_r, self.qty_c, self.offblocks)
        self.assertEqual(len(node), 0)

    def test_ctr_tank_fuel(self):
        self.qty_c.array += 1000

        node = MainFuelQuantityOffBlockWithFuelInCenterTank()
        node.derive(self.qty_l, self.qty_r, self.qty_c, self.offblocks)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 32400 + 33700)
        self.assertEqual(node[0].index, 3)


##############################################################################
# Groundspeed


class TestGroundspeedWithGearOnGroundMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWithGearOnGroundMax
        self.operational_combinations = [('Groundspeed Signed', 'Gear On Ground', 'Grounded')]
        self.function = max_value

    def test_derive_basic(self):
        spd = P('Groundspeed Signed', array = np.ma.arange(100, 0, -10))
        gog = M('Gear On Ground', array=np.ma.array([0]*5 + [1]*5), values_mapping = {0: 'Air', 1: 'Ground'})
        groundeds = buildsection('Grounded', 6, 10)

        node = self.node_class()
        node.derive(spd, gog, groundeds)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=6, value=40.0,
            name='Groundspeed With Gear On Ground Max'))


class TestGroundspeedWhileTaxiingStraightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWhileTaxiingStraightMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedInStraightLineDuringTaxiInMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedInStraightLineDuringTaxiInMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxi In', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedInStraightLineDuringTaxiOutMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedInStraightLineDuringTaxiOutMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxi Out', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedWhileTaxiingTurnMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWhileTaxiingTurnMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedInTurnDuringTaxiInMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedInTurnDuringTaxiInMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxi In', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedInTurnDuringTaxiOutMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedInTurnDuringTaxiOutMax
        self.operational_combinations = [('Groundspeed Signed', 'Taxi Out', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedDuringRejectedTakeoffMax(unittest.TestCase):

    def test_can_operate(self):
        expected = [
            ('Acceleration Longitudinal', 'Rejected Takeoff'),
            ('Groundspeed Signed', 'Rejected Takeoff'),
            ('Acceleration Longitudinal', 'Groundspeed Signed', 'Rejected Takeoff')
        ]
        opts = GroundspeedDuringRejectedTakeoffMax.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive(self):
        gspd = P('Groundspeed Signed',
                 np.ma.array([0]*20 + list(range(20)) + list(range(20, 1, -1)) + list(range(80))))
        rto = buildsection('Rejected Takeoff', 22, 40)
        gspd_rto = GroundspeedDuringRejectedTakeoffMax()
        gspd_rto.derive(accel=None, gnd_spd=gspd, rtos=rto)
        self.assertEqual(len(gspd_rto), 1)
        self.assertEqual(gspd_rto[0].index, 40)
        self.assertEqual(gspd_rto[0].value, 20)

    def test_no_gspd(self):
        sinewave = np.ma.sin(np.arange(0, 3.14*2, 0.04))*0.4
        testwave = [0]*150+sinewave.tolist()+[0]*193 # To match array sizes
        accel = P('Acceleration Longitudinal',
                  testwave, frequency=4.0, offset=0.2)
        gnds = buildsections('Grounded', [0, 125], [400, 500])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        trh = buildsections('Takeoff Runway Heading', [0, 125])
        # Create RTO here to ensure it operates as expected
        rto = RejectedTakeoff()
        rto.get_derived((accel, eng_running, gnds, None, None, trh, A(
            'Segment Type', 'START_AND_STOP')))
        # The data passes 0.1g on the 6th and 72nd samples of the sine wave.
        # With the 24 sample offset and 4Hz sample rate this gives an RTO
        # section thus:
        ##rto = buildsection('Rejected Takeoff', (24+6)/4.0, (24+72)/4.0)
        gspd_rto = GroundspeedDuringRejectedTakeoffMax()
        gspd_rto.get_derived((accel, None, rto))
        # The groundspeed should match the integration. Done crudely here for plotting when developing the routine.
        #expected = np.cumsum(testwave)*32.2*0.592/4

        self.assertAlmostEqual(gspd_rto[0].value, 91.6, 1)
        self.assertEqual(gspd_rto[0].index, 229)


class TestGroundspeedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GroundspeedAtLiftoff
        self.operational_combinations = [('Groundspeed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GroundspeedAtTouchdown
        self.operational_combinations = [('Groundspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedVacatingRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GroundspeedVacatingRunway

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=helicopter), [])
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 2)
        self.assertIn('Groundspeed Signed', opts[0])
        self.assertIn('Landing Turn Off Runway', opts[0])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedAtTOGA(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedAtTOGA
        self.operational_combinations = [('Takeoff And Go Around', 'Groundspeed', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedWithThrustReversersDeployedMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWithThrustReversersDeployedMin
        self.operational_combinations = [
            ('Groundspeed', 'Thrust Reversers', 'Eng (*) EPR Max',
             'Eng (*) N1 Max', 'Landing'),
            ('Groundspeed', 'Thrust Reversers', 'Eng (*) EPR Max', 'Landing'),
            ('Groundspeed', 'Thrust Reversers', 'Eng (*) N1 Max', 'Landing')]

    def test_derive_basic(self):
        spd=P('Groundspeed True', array = np.ma.arange(100, 0, -10))
        tr=M('Thrust Reversers',
             array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
             values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        # half the frequency of spd
        n1=P('Eng (*) N1 Max', frequency=0.5,
             array=np.ma.array([40] * 2 + [70] * 3))
        landings=buildsection('Landing', 2, 9)
        node = GroundspeedWithThrustReversersDeployedMin()
        node.derive(spd, tr, None, n1, landings)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=7, value=30.0,
            name='Groundspeed With Thrust Reversers Deployed Min'))

    def test_derive_with_epr(self):
        spd = P('Groundspeed True', array = np.ma.arange(100, 0, -10))
        tr = M('Thrust Reversers',
               array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
               values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        # half the frequency of spd
        epr = P('Eng (*) EPR Max', frequency=0.5,
                array=np.ma.array([1.0] * 2 + [1.26] * 2 + [1.0] * 1))
        landings=buildsection('Landing', 2, 9)
        node = GroundspeedWithThrustReversersDeployedMin()
        node.derive(spd, tr, epr, None, landings, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=6, value=40.0,
            name='Groundspeed With Thrust Reversers Deployed Min'))

    def test_derive_with_epr_n1_equal_freq(self):
        spd = P('Groundspeed True', array = np.ma.arange(100, 0, -10))
        tr = M('Thrust Reversers',
               array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
               values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        # half the frequency of spd
        epr = P('Eng (*) EPR Max', frequency=0.5,
                array=np.ma.array([1.0] * 2 + [1.26] * 2 + [1.0] * 1))
        n1=P('Eng (*) N1 Max', frequency=4, # Frequency set to 4 as the derived N1 freq should be ignored anyway
             array=np.ma.array([40] * 16 + [70] * 24))
        n1_recorded=P('Eng (1) N1', frequency=0.5,
                      array=np.ma.array([40] * 2 + [70] * 3))
        landings=buildsection('Landing', 2, 9)
        node = GroundspeedWithThrustReversersDeployedMin()
        node.derive(spd, tr, epr, n1, landings, n1_recorded)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=6, value=40.0,
            name='Groundspeed With Thrust Reversers Deployed Min'))

    def test_derive_with_epr_freq_higher(self):
        spd = P('Groundspeed True', array = np.ma.arange(100, 0, -10))
        tr = M('Thrust Reversers',
               array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
               values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        # half the frequency of spd
        epr = P('Eng (*) EPR Max', frequency=0.5,
                array=np.ma.array([1.0] * 2 + [1.26] * 2 + [1.0] * 1))
        n1=P('Eng (*) N1 Max', frequency=4, # Frequency set to 4 as the derived N1 freq should be ignored anyway
             array=np.ma.array([40] * 16 + [70] * 24))
        n1_recorded=P('Eng (1) N1', frequency=0.25,
                      array=np.ma.array([40] * 2 + [70] * 3))
        landings=buildsection('Landing', 2, 9)
        node = GroundspeedWithThrustReversersDeployedMin()
        node.derive(spd, tr, epr, n1, landings, n1_recorded)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=6, value=40.0,
            name='Groundspeed With Thrust Reversers Deployed Min'))

    def test_derive_with_n1_freq_higher(self):
        spd = P('Groundspeed True', array = np.ma.arange(100, 0, -10))
        tr = M('Thrust Reversers',
               array=np.ma.array([0] * 3 + [1] + [2] * 4 + [1,0]),
               values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'})
        # half the frequency of spd
        epr = P('Eng (*) EPR Max', frequency=0.25,
                array=np.ma.array([1.0] * 2 + [1.26] * 2 + [1.0] * 1))
        n1=P('Eng (*) N1 Max', frequency=4, # Frequency set to 4 as the derived N1 freq should be ignored anyway
             array=np.ma.array([40] * 16 + [70] * 24))
        n1_recorded=P('Eng (1) N1', frequency=0.5,
                      array=np.ma.array([40] * 2 + [70] * 3))
        landings=buildsection('Landing', 2, 9)
        node = GroundspeedWithThrustReversersDeployedMin()
        node.derive(spd, tr, epr, n1, landings, n1_recorded)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(
            index=7, value=30.0,
            name='Groundspeed With Thrust Reversers Deployed Min'))

class TestGroundspeedStabilizerOutOfTrimDuringTakeoffMax(unittest.TestCase,
                                                         NodeTest):
    def setUp(self):
        self.node_class = GroundspeedStabilizerOutOfTrimDuringTakeoffMax
        self.operational_combinations = []
        # FIXME: can_operate uses the Family and Series
        #    ('Groundspeed', 'Stabilizer', 'Takeoff Roll', 'Model', 'Series', 'Family')]

    def test_derive(self):
        array = np.arange(10) + 100
        array = np.ma.concatenate((array[::-1], array))
        gspd = P('Groundspeed', array)

        array = np.arange(20, 100, 4) * 0.1
        stab = P('Stabilizer', array)

        phase = S(frequency=1)
        phase.create_section(slice(0, 20))

        model = A(name='Model', value=None)
        series = A(name='Series', value='B737-600')
        family = A(name='Family', value='B737 NG')

        node = self.node_class()
        node.derive(gspd, stab, phase, model, series, family)
        self.assertEqual(
            node,
            KPV(self.node_class.get_name(),
                items=[KeyPointValue(name=self.node_class.get_name(),
                                     index=0.0, value=109.0)])
        )


class TestGroundspeedSpeedbrakeHandleDuringTakeoffMax(unittest.TestCase,
                                                      NodeTest):
    def setUp(self):
        self.node_class = GroundspeedSpeedbrakeHandleDuringTakeoffMax
        self.operational_combinations = [
            ('Groundspeed', 'Speedbrake Handle', 'Speedbrake Selected',
             'Takeoff Roll Or Rejected Takeoff')]

    def test_derive_speedbrake_deployed(self):
        array = np.arange(10) + 100
        array = np.ma.concatenate((array[::-1], array))
        gspd = P('Groundspeed', array)

        array = 1 + np.arange(0, 20, 2) * 0.1
        array = np.ma.concatenate((array[::-1], array))
        spdbrk = P('Speedbrake Handle', array)

        spdbrk_selected = M(name='Speedbrake Selected',
                            array=np.ma.array([2]*20),
                            values_mapping = {
                                0: 'Stowed',
                                1: 'Armed/Cmd Dn',
                                2: 'Deployed/Cmd Up',})

        phase = S(frequency=1)
        phase.create_section(slice(0, 20))

        node = self.node_class()
        node.derive(gspd, spdbrk, spdbrk_selected, phase)
        self.assertEqual(
            node,
            KPV(self.node_class.get_name(),
                items=[KeyPointValue(name=self.node_class.get_name(),
                                     index=0.0, value=109.0)])
        )

    def test_derive_speedbrake_not_deployed(self):
        array = np.arange(10) + 100
        array = np.ma.concatenate((array[::-1], array))
        gspd = P('Groundspeed', array)

        array = 1 + np.arange(0, 20, 2) * 0.1
        array = np.ma.concatenate((array[::-1], array))
        spdbrk = P('Speedbrake Handle', array)

        spdbrk_selected = M(name='Speedbrake Selected',
                            array=np.ma.array([0]*10 + [1]*10),
                            values_mapping = {
                                0: 'Stowed',
                                1: 'Armed/Cmd Dn',
                                2: 'Deployed/Cmd Up',})

        phase = S(frequency=1)
        phase.create_section(slice(0, 20))

        node = self.node_class()
        node.derive(gspd, spdbrk, spdbrk_selected, phase)
        self.assertEqual(
            node,
            KPV(self.node_class.get_name(),
                items=[])
        )



class TestGroundspeedSpeedbrakeDuringTakeoffMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = GroundspeedSpeedbrakeDuringTakeoffMax
        self.operational_combinations = [
            ('Groundspeed', 'Speedbrake', 'Takeoff Roll Or Rejected Takeoff')]

    def test_derive(self):
        array = np.arange(10) + 100
        array = np.ma.concatenate((array[::-1], array))
        gspd = P('Groundspeed', array)

        array = 20 + np.arange(5, 25, 2)
        array = np.ma.concatenate((array[::-1], array))
        stab = P('Speedbrake', array)

        phase = S(frequency=1)
        phase.create_section(slice(0, 20))

        node = self.node_class()
        node.derive(gspd, stab, phase)
        self.assertEqual(
            node,
            KPV(self.node_class.get_name(),
                items=[KeyPointValue(name=self.node_class.get_name(),
                                     index=0.0, value=109.0)])
        )


class TestGroundspeedFlapChangeDuringTakeoffMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedFlapChangeDuringTakeoffMax
        self.operational_combinations = [('Groundspeed', 'Flap',
                                          'Takeoff Roll Or Rejected Takeoff')]

    def test_derive(self):
        array = np.ma.arange(10) + 100
        array = np.ma.concatenate((array[::-1], array))
        gnd_spd = P(name='Groundspeed', array=array)

        array = np.ma.array([0] * 5 + [10] * 10 + [0] * 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap = M(name='Flap', array=array, values_mapping=mapping)

        phase = S(frequency=1)
        phase.create_section(slice(0, 20))

        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(gnd_spd, flap, phase)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=15.0, value=105.0, name=name),
        ]))



##############################################################################
# Law


class TestAlternateLawDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AlternateLawDuration
        self.operational_combinations = [
            ('Alternate Law', 'Airborne'),
            ('Pitch Alternate Law', 'Airborne'),
            ('Roll Alternate Law', 'Airborne'),
            ('Alternate Law', 'Pitch Alternate Law', 'Roll Alternate Law', 'Airborne'),
        ]

    def test_derive_basic(self):
        array = np.ma.array([0] * 7 + [1] * 3 + [0] * 6)
        mapping = {0: '-', 1: 'Engaged'}
        alternate_law = M('Alternate Law', array, values_mapping=mapping)
        pitch_alternate_law = M('Pitch Alternate Law', np.roll(array, 2), values_mapping=mapping)
        roll_alternate_law = M('Roll Alternate Law', np.roll(array, 4), values_mapping=mapping)
        roll_alternate_law.array[0] = 'Engaged'
        airborne = buildsection('Airborne', 2, 20)
        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=7, value=7),
        ])
        node = self.node_class()
        node.derive(alternate_law, pitch_alternate_law, roll_alternate_law, airborne)
        self.assertEqual(node, expected)


class TestDirectLawDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DirectLawDuration
        self.operational_combinations = [
            ('Direct Law', 'Airborne'),
            ('Pitch Direct Law', 'Airborne'),
            ('Roll Direct Law', 'Airborne'),
            ('Direct Law', 'Pitch Direct Law', 'Roll Direct Law', 'Airborne'),
        ]

    def test_derive_basic(self):
        array = np.ma.array([0] * 7 + [1] * 3 + [0] * 6)
        mapping = {0: '-', 1: 'Engaged'}
        direct_law = M('Direct Law', array, values_mapping=mapping)
        pitch_direct_law = M('Pitch Direct Law', np.roll(array, 2), values_mapping=mapping)
        pitch_direct_law.array[0] = 'Engaged'
        roll_direct_law = M('Roll Direct Law', np.roll(array, 4), values_mapping=mapping)
        airborne = buildsection('Airborne', 2, 20)
        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=7, value=7),
        ])
        node = self.node_class()
        node.derive(direct_law, pitch_direct_law, roll_direct_law, airborne)
        self.assertEqual(node, expected)


##############################################################################
# Pitch


class TestPitchAfterFlapRetractionMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAfterFlapRetractionMax
        self.operational_combinations = [
            ('Flap Lever', 'Pitch', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Pitch', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Pitch', 'Airborne'),
        ]

    def test_derive(self):
        pitch = P(
            name='Pitch',
            array=np.ma.repeat(range(7, 0, -1), 5) * 0.1,
        )
        airborne = buildsection('Airborne', 2, 28)
        name = self.node_class.get_name()

        array = np.ma.repeat((5, 1, 0, 15, 25, 30), 5)
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(pitch, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.5, name=name),
        ]))

        array = np.ma.repeat((5, 2, 1, 15, 25, 30), 5)
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(pitch, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.5, name=name),
        ]))


class TestPitchAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = PitchAtLiftoff
        self.operational_combinations = [('Pitch', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitchAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = PitchAtTouchdown
        self.operational_combinations = [('Pitch', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitchAt35FtDuringClimb(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAt35FtDuringClimb
        self.operational_combinations = [('Pitch', 'Altitude AAL', 'Initial Climb')]
        self.can_operate_kwargs = {'ac_type': aeroplane}

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchTakeoffMax
        self.operational_combinations = [('Pitch', 'Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')



class TestPitchAbove1000FtMin(unittest.TestCase):

    def can_operate(self):
        opts = PitchAbove1000FtMin.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Altitude AAL')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 10, 12, 13, 8, 14, 6])
        aal = P('Altitude AAL',
                array=[100, 200, 700, 1010, 4000, 1200, 1100, 900, 800])
        node = PitchAbove1000FtMin()
        node.derive(pitch, aal)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 8)


class TestPitchAbove1000FtMax(unittest.TestCase):

    def can_operate(self):
        opts = PitchAbove1000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Altitude AAL')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 10, 12, 13, 8, 14, 6, 20])
        aal = P('Altitude AAL',
                array=[100, 200, 700, 1010, 4000, 1200, 1100, 900, 800])
        node = PitchAbove1000FtMax()
        node.derive(pitch, aal)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 14)


class TestPitch35ToClimbAccelerationStartMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35ToClimbAccelerationStartMax
        self.operational_combinations = [('Pitch', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Pitch 35 To Climb Acceleration Start Max', items=[
            KeyPointValue(name='Pitch 35 To Climb Acceleration Start Max', index=3, value=7),
        ]))


class TestPitch35ToClimbAccelerationStartMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35ToClimbAccelerationStartMin
        self.operational_combinations = [('Pitch', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(opts, self.operational_combinations)

    def test_derive(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch, climb, climb_accel_start)

        self.assertEqual(node, KPV('Pitch 35 To Climb Acceleration Start Min', items=[
            KeyPointValue(name='Pitch 35 To Climb Acceleration Start Min', index=1.4, value=2.8),
        ]))


class TestPitch35To400FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35To400FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([100, 101, 102, 103, 700, 105, 104, 103, 102]),
        )
        climb = buildsection('Climb', 0, 4)
        node = Pitch35To400FtMax()
        node.derive(pitch, alt_aal, climb)
        self.assertEqual(node, KPV('Pitch 35 To 400 Ft Max', items=[
            KeyPointValue(name='Pitch 35 To 400 Ft Max', index=3, value=7),
        ]))


class TestPitch35To400FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35To400FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitch400To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch400To1000FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch400To1000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch400To1000FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch1000To500FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL For Flight Phases', 'Descent'), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch1000To500FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = min_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL', 'Descent'), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To50FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch500To50FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL', 'Descending'), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch500To20FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To7FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch500To7FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 7), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To7FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch500To7FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 7), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch7FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch7FtToTouchdownMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (7, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch50FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch50FtToTouchdownMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Touchdown', 'Altitude AGL', 'Descending'), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch20FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch20FtToTouchdownMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        heli_opts = ('Pitch', 'Altitude AGL', 'Touchdown', 'Aircraft Type')
        aero_opts = ('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown', 'Aircraft Type')
        self.assertTrue(self.node_class.can_operate(heli_opts, ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(aero_opts, ac_type=aeroplane))

    def test_derive(self):
        alt = P('Altitude AGL', np.ma.array(list(range(90, 0, -1))+[0]*10))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown')])

        node = self.node_class()
        node.derive(pitch, None, alt, tdwns, ac_type=helicopter)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 89)
        self.assertAlmostEqual(node[0].value, -3.787, places=3)

        node = self.node_class()
        node.derive(pitch, alt, None, tdwns, ac_type=aeroplane)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 89)
        self.assertAlmostEqual(node[0].value, -3.787, places=3)


class TestPitch20FtToTouchdownMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch20FtToTouchdownMin

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        heli_opts = ('Pitch', 'Altitude AGL', 'Touchdown', 'Aircraft Type')
        aero_opts = ('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown', 'Aircraft Type')
        self.assertTrue(self.node_class.can_operate(heli_opts, ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(aero_opts, ac_type=aeroplane))

    def test_derive(self):
        alt = P('Altitude AGL', np.ma.array(list(range(90, 0, -1))+[0]*10))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown')])

        node = self.node_class()
        node.derive(pitch, None, alt, tdwns, ac_type=helicopter)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)

        node = self.node_class()
        node.derive(pitch, alt, None, tdwns, ac_type=aeroplane)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)


class TestPitch500To50FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch500To50FtMin

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL', 'Descending'), ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AGL', 'Descending'), ac_type=aeroplane))

    def test_derive(self):
        alt = P('Altitude AAL For Flight Phases', np.ma.arange(0, 5000, 50))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        name = 'Descending'
        section = Section(name, slice(0, 150), 0, 150)
        descending = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, alt, descending)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertAlmostEqual(node[0].value, -0.855, places=3)


class TestPitch7FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch7FtToTouchdownMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (7, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchCyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchCyclesDuringFinalApproach
        self.operational_combinations = [('Pitch', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchDisconnectDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchDisconnectDuration

    def test_derive(self):
        pitch_disc = M(
            name='Pitch Disconnect',
            array=np.ma.array([0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Disconnect'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(pitch_disc)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 3)


class TestPitchDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchDuringGoAroundMax
        self.operational_combinations = [('Pitch', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchWhileAirborneMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Airborne')])

    def test_derive(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        name = 'Airborne'
        section = Section(name, slice(1, 8), 1, 8)
        airborne = SectionNode(name, items=[section])
        node = self.node_class()
        node.derive(pitch, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 4)
        self.assertEqual(node[0].value, 9)


class TestPitchWhileAirborneMin(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchWhileAirborneMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Airborne')])

    def test_derive(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        name = 'Airborne'
        section = Section(name, slice(1, 8), 1, 7.1)
        airborne = SectionNode(name, items=[section])
        node = self.node_class()
        node.derive(pitch, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 1)
        self.assertEqual(node[0].value, 2)


class TestPitchTouchdownTo60KtsAirspeedMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = PitchTouchdownTo60KtsAirspeedMax
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Pitch', 'Airspeed', 'Touchdown')]

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'Pitch Touchdown To 60 Kts Airspeed Max')
        self.assertEqual(node.units, ut.DEGREE)

    def test_derive(self):
        pitch = P('Pitch', np.ma.array([0.7, 1.1, 0.2, -0.8, -1.2,
                                        -0.7, -0.1, 0.1, 0.5, -0.5,
                                             -0.1, -0.1, -0.1, 1.0, 0.0]))

        airspeed = P('Airspeed', np.ma.array(np.linspace(110, 47, 15)))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(3, 'Touchdown')])

        node = self.node_class()
        node.derive(pitch, airspeed, touchdown)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8)
        self.assertEqual(node[0].value, 0.5)


class TestPitchAboveFL200Max(unittest.TestCase):

    def can_operate(self):
        opts = PitchAboveFL200Max.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Altitude STD Smoothed')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 10, 12, 13, 8, 14, 6, 5])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = PitchAboveFL200Max()
        node.derive(pitch, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 13)


class TestPitchAboveFL200Min(unittest.TestCase):

    def can_operate(self):
        opts = PitchAboveFL200Min.get_operational_combinations()
        self.assertEqual(opts, [('Pitch', 'Altitude STD Smoothed')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 10, 12, 13, 8, 14, 6, 5])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = PitchAboveFL200Min()
        node.derive(pitch, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 8)


##############################################################################
# Pitch Rate


class TestPitchRate35To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate35To1000FtMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate35ToClimbAccelerationStartMax(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchRate35ToClimbAccelerationStartMax
        self.operational_combinations = [('Pitch Rate', 'Initial Climb', 'Climb Acceleration Start')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch_rate = P(
            name='Pitch Rate',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        climb = buildsection('Initial Climb', 1.4, 8)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(3, 'Climb Acceleration Start')])

        node = self.node_class()
        node.derive(pitch_rate, climb, climb_accel_start)

        self.assertEqual(node, KPV('Pitch Rate 35 To Climb Acceleration Start Max', items=[
            KeyPointValue(name='Pitch Rate 35 To Climb Acceleration Start Max', index=3, value=7),
        ]))


class TestPitchRate20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = PitchRate20FtToTouchdownMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = PitchRate20FtToTouchdownMin
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate2DegPitchTo35FtMax
        self.operational_combinations = [('Pitch Rate', '2 Deg Pitch To 35 Ft')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate2DegPitchTo35FtMin
        self.operational_combinations = [('Pitch Rate', '2 Deg Pitch To 35 Ft')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRateWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchRateWhileAirborneMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Pitch Rate', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 50)
        pitch = P(
            name='Pitch Rate',
            array=-x*np.sin(x),
        )
        name = 'Airborne'
        section = Section(name, slice(10, 41), 9.5, 40.5)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 39)
        self.assertAlmostEqual(node[0].value, -7.915, places=3)


class TestPitchRateTouchdownTo60KtsAirspeedMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = PitchRateTouchdownTo60KtsAirspeedMax
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Pitch Rate', 'Airspeed', 'Touchdown')]

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'Pitch Rate Touchdown To 60 Kts Airspeed Max')
        self.assertEqual(node.units, ut.DEGREE_S)

    def test_derive(self):
        pitch = P('Pitch Rate', np.ma.array([0.7, 1.1, 0.2, -0.8, -1.2,
                                             -0.7, -0.1, 0.1, 0.5, -0.5,
                                             -0.1, -0.1, -0.1, 1.0, 0.0]))

        airspeed = P('Airspeed', np.ma.array(np.linspace(110, 47, 15)))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(3, 'Touchdown')])

        node = self.node_class()
        node.derive(pitch, airspeed, touchdown)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8)
        self.assertEqual(node[0].value, 0.5)
##############################################################################
# Rate of Climb


class TestRateOfClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfClimbMax
        self.operational_combinations = [('Vertical Speed', 'Climbing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimb35ToClimbAccelerationStartMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfClimb35ToClimbAccelerationStartMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Initial Climb', 'Climb Acceleration Start')])

    def test_derive_basic(self):
        roc_array = np.ma.concatenate(([25]*19, [43, 62, 81, 100, 112, 24, 47, 50, 12, 37, 27, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        climb = buildsection('Initial Climb', 1.4, 28)
        climb_accel_start = KTI('Climb Acceleration Start', items=[KeyTimeInstance(25, 'Touchdown')])

        node = self.node_class()
        node.derive(vert_spd, climb, climb_accel_start)

        expected = KPV('Rate Of Climb 35 To Climb Acceleration Start Min', items=[
            KeyPointValue(name='Rate Of Climb 35 To Climb Acceleration Start Min', index=24, value=24),
        ])
        self.assertEqual(node, expected)


class TestRateOfClimb35To1000FtMin(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimb35To1000FtMin.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Initial Climb')])

    def test_derive(self):
        ##array = np.ma.concatenate((np.ma.arange(0, 500, 25), np.ma.arange(500, 1000, 100), [1050, 950, 990], [1100]*5))
        ##array = np.ma.concatenate((array, array[::-1]))
        ##alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([25]*19, [43, 62, 81, 100, 112, 62, 47, 50, 12, 37, 27, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        climb = buildsection('Initial Climb', 1.4, 28)

        node = RateOfClimb35To1000FtMin()
        node.derive(vert_spd, climb)

        expected = KPV('Rate Of Climb 35 To 1000 Ft Min', items=[
            KeyPointValue(name='Rate Of Climb 35 To 1000 Ft Min', index=27, value=12),
        ])
        self.assertEqual(node, expected)


class TestRateOfClimbBelow10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimbBelow10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Airborne')])

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 5000, 250), np.arange(5000, 10000, 1000), [10500, 9500, 9900], np.ones(5) * 11000))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude STD Smoothed', array)
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1 - roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)
        airs = buildsection('Airborne', 1, 200)
        node = RateOfClimbBelow10000FtMax()
        node.derive(vert_spd, alt, airs)

        expected = KPV('Rate Of Climb Below 10000 Ft Max', items=[
            KeyPointValue(name='Rate Of Climb Below 10000 Ft Max', index=23, value=1125),
        ])
        self.assertEqual(node, expected)

    def test_airborne_restriction(self):
        array = np.ma.concatenate((np.arange(0, 5000, 250), np.arange(5000, 10000, 1000), [10500, 9500, 9900], np.ones(5) * 11000))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude STD Smoothed', array)
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1-roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)
        airs = buildsection('Airborne', 1, 2)
        node = RateOfClimbBelow10000FtMax()
        node.derive(vert_spd, alt, airs)

        expected = KPV('Rate Of Climb Below 10000 Ft Max', items=[
            KeyPointValue(name='Rate Of Climb Below 10000 Ft Max', index=1, value=250),
        ])
        self.assertEqual(node, expected)


class TestRateOfClimbDuringGoAroundMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimbDuringGoAroundMax.get_operational_combinations()
        self.assertEqual(opts, [
            ('Vertical Speed', 'Go Around And Climbout')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimbAtHeightBeforeLevelOff(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfClimbAtHeightBeforeLevelFlight

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,
                         [('Vertical Speed',
                           'Altitude Before Level Flight When Climbing')])

    def test_derive(self):
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125,
                                                           625, 475, 500, 125, 375,
                                                           275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1-roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)
        heights = AltitudeBeforeLevelFlightWhenClimbing(
            'Altitude Before Level Flight When Climbing',
            items=[KeyTimeInstance(22,
                                   '2000 Ft Before Level Flight Climbing'),
                   KeyTimeInstance(24,
                                   '1000 Ft Before Level Flight Climbing'),
                   ])
        node = self.node_class()
        node.derive(vert_spd, heights)

        expected = KPV('Rate Of Climb At Height Before Level Flight', items=[
            KeyPointValue(name='Rate Of Climb At 2000 Ft Before Level Off',
                          index=22,
                          value=1000),
            KeyPointValue(name='Rate Of Climb At 1000 Ft Before Level Off',
                          index=24,
                          value=625),
        ])
        self.assertEqual(node, expected)


##############################################################################
# Rate of Descent


class TestRateOfDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentMax
        self.operational_combinations = [('Vertical Speed', 'Descending')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentTopOfDescentTo10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescentTopOfDescentTo10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescentAbove3000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescentAbove3000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Descent')])

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 3500, 350), np.ones(10) * 3500))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375], np.zeros(10)))
        roc_array = np.ma.concatenate((1 - roc_array, roc_array[::-1]))
        vert_spd = P('Vertical Speed', -roc_array)
        airs = buildsection('Descent', 30, 39)
        node = RateOfDescentAbove3000FtMax()
        node.derive(vert_spd, alt, airs)

        expected = KPV('Rate Of Descent Above 3000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent Above 3000 Ft Max', index=30, value=-375),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescentBelow10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescentBelow10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Descent')])

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 5000, 250), np.arange(5000, 10000, 1000), [10500, 9500, 9900], np.ones(5) * 11000))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude STD Smoothed', array)
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1 - roc_array[::-1]))
        vert_spd = P('Vertical Speed', -roc_array)
        airs = buildsection('Descent', 1, 200)
        node = RateOfDescentBelow10000FtMax()
        node.derive(vert_spd, alt, airs)

        expected = KPV('Rate Of Descent Below 10000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent Below 10000 Ft Max', index=23, value=-1125),
        ])
        self.assertEqual(node, expected)

    def test_airborne_restriction(self):
        array = np.ma.concatenate((np.arange(0, 5000, 250), np.arange(5000, 10000, 1000), [10500, 9500, 9900], np.ones(5) * 11000))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude STD Smoothed', array)
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1 - roc_array[::-1]))
        vert_spd = P('Vertical Speed', -roc_array)
        airs = buildsection('Descent', 1, 2)
        node = RateOfDescentBelow10000FtMax()
        node.derive(vert_spd, alt, airs)

        expected = KPV('Rate Of Descent Below 10000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent Below 10000 Ft Max', index=1, value=-250),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent10000To5000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent10000To5000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent5000To3000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent5000To3000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent3000To2000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent3000To2000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases')])

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 1500, 100), np.arange(1500, 3000, 200), [3050, 2910, 2990], np.ones(5) * 3150))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(14) * 100, [125, 150, 175, 200, 200, 200, 200, 187, 87, 72, 62, 25, 75, 40, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        node = RateOfDescent3000To2000FtMax()
        node.derive(vert_spd, alt)

        expected = KPV('Rate Of Descent 3000 To 2000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent 3000 To 2000 Ft Max', index=41, value=-200),
        ])
        self.assertEqual(node, expected)



class TestRateOfDescent2000To1000FtMax(unittest.TestCase):


    def test_can_operate(self):
        opts = RateOfDescent2000To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases')])

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 500, 25), np.arange(500, 2000, 100), [2050, 1850, 1990], np.ones(5) * 2150))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(19) * 25, [43, 62, 81, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 112, 37, 47, 62, 25, 75, 40, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        node = RateOfDescent2000To1000FtMax()
        node.derive(vert_spd, alt)

        expected = KPV('Rate Of Descent 2000 To 1000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent 2000 To 1000 Ft Max', index=48, value=-25),
            KeyPointValue(name='Rate Of Descent 2000 To 1000 Ft Max', index=52, value=-112),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent1000To500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent1000To500FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed', 'Altitude AGL', 'Descent'), ac_type=helicopter))

    @unittest.SkipTest
    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 500, 25), np.arange(500, 1000, 100), [1050, 950, 990], np.ones(5) * 1090))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(19) * 25, [43, 62, 81, 100, 112, 62, 47, 47, 10, 35, 25, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Final Approach', 37, 63)

        node = RateOfDescent1000To500FtMax()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 1000 To 500 Ft Max', items=[
            KeyPointValue(index=39.0, value=-47.0, name='Rate Of Descent 1000 To 500 Ft Max'),
            KeyPointValue(index=42.0, value=-112.0, name='Rate Of Descent 1000 To 500 Ft Max')
        ])
        self.assertEqual(node, expected)

    def test_derive_helicopter(self):
        array = np.ma.concatenate((np.arange(0, 500, 25), np.arange(500, 1000, 100), [1050, 950, 990], np.ones(5) * 1090))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array, frequency=0.25)
        roc_array = np.ma.concatenate((np.ones(19) * 25, [43, 62, 81, 100, 112, 62, 47, 47, 10, 35, 25, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array, frequency=0.25)
        name = 'Descent'
        section = Section(name, slice(37, 63), 37, 63)
        descents = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(vert_spd, helicopter, None, None, alt, descents)

        expected = KPV('Rate Of Descent 1000 To 500 Ft Max', items=[
            KeyPointValue(index=39.0, value=-47.0, name='Rate Of Descent 1000 To 500 Ft Max'),
            KeyPointValue(index=42.0, value=-112.0, name='Rate Of Descent 1000 To 500 Ft Max')
        ])
        self.assertEqual(node, expected)

    def test_derive_helicopter_short(self):
        array = np.ma.concatenate((np.ma.arange(0, 500, 25), np.ma.arange(500, 1000, 100), [1050, 950, 990], np.ones(5) * 1090))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(19) * 25, [43, 62, 81, 100, 112, 62, 47, 47, 10, 35, 25, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)
        name = 'Final Approach'
        section = Section(name, slice(37, 63), 37, 63)
        descents = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(vert_spd, helicopter, None, None, alt, descents)

        expected = []
        self.assertEqual(node, expected)


class TestRateOfDescent500To50FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent500To50FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed', 'Altitude AGL', 'Descending'), ac_type=helicopter))

    @unittest.SkipTest
    def test_derive_aeroplane(self):
        array = np.ma.concatenate((np.arange(0, 50, 25), np.arange(50, 500, 100), [550, 450, 540], np.ones(5) * 590))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(2) * 25, [62, 81, 100, 100, 50, 47, 35, 10, 35, 12, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Final Approach', 19, 27)

        node = RateOfDescent500To50FtMax()
        node.derive(vert_spd, aeroplane, alt, descents, None, None)

        expected = KPV('Rate Of Descent 500 To 50 Ft Max', items=[
            KeyPointValue(index=21.0, value=-35.0, name='Rate Of Descent 500 To 50 Ft Max'),
            KeyPointValue(index=24.0, value=-100.0, name='Rate Of Descent 500 To 50 Ft Max')
        ])
        self.assertEqual(node, expected)

    def test_derive_helicopter(self):
        array = np.ma.concatenate((np.arange(0, 50, 25), np.arange(50, 500, 100), [550, 450, 540], np.arange(5) * 590))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AGL', array, frequency=0.25)
        roc_array = np.ma.concatenate((np.ones(2) * 25, [62, 81, 100, 100, 50, 47, 35, 10, 35, 12, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array, frequency=0.25)
        name = 'Descending'
        section = Section(name, slice(19, 27), 19, 27)
        descents = SectionNode(name, items=[section], frequency=0.25)

        node = self.node_class()
        node.derive(vert_spd, helicopter, None, None, alt, descents)

        expected = KPV('Rate Of Descent 500 To 50 Ft Max', items=[
            KeyPointValue(index=24.0, value=-100.0, name='Rate Of Descent 500 To 50 Ft Max')
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent50FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent50FtToTouchdownMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed Inertial', 'Touchdown', 'Altitude AGL',), ac_type=helicopter))

    def test_derive_aeroplane(self):
        array = np.ma.concatenate((np.ma.arange(0, 50, 5), [55, 45, 54], np.ones(5) * 59))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(8) * 5, [6, 2, 3, 3, 1, 3, 1, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        roc_array[33] = -26
        vert_spd = P('Vertical Speed Inertial', roc_array)

        touch_down = KTI('Touchdown', items=[KeyTimeInstance(34, 'Touchdown')])

        node = RateOfDescent50FtToTouchdownMax()
        node.derive(vert_spd, touch_down, alt, None)

        expected = KPV('Rate Of Descent 50 Ft To Touchdown Max', items=[
            KeyPointValue(name='Rate Of Descent 50 Ft To Touchdown Max', index=33, value=-26),
        ])
        self.assertEqual(node, expected)

    def test_derive_helicopter(self):
        array = np.ma.concatenate((np.ma.arange(0, 50, 5), [55, 45, 54], np.ones(5) * 59))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(8) * 5, [6, 2, 3, 3, 1, 3, 1, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        roc_array[33] = -26
        vert_spd = P('Vertical Speed Inertial', roc_array)

        touch_down = KTI('Touchdown', items=[KeyTimeInstance(34, 'Touchdown')])

        node = self.node_class()
        node.derive(vert_spd, touch_down, None, alt)

        expected = KPV('Rate Of Descent 50 Ft To Touchdown Max', items=[
            KeyPointValue(name='Rate Of Descent 50 Ft To Touchdown Max', index=33, value=-26),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescentAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentAtTouchdown
        self.operational_combinations = [('Vertical Speed Inertial', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentDuringGoAroundMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimbDuringGoAroundMax.get_operational_combinations()
        self.assertEqual(opts, [
            ('Vertical Speed', 'Go Around And Climbout')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentBelow80KtsMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescentBelow80KtsMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Airspeed', 'Descending')])

    def test_derive(self,):
        x = np.linspace(0, 10, 62)
        vrt_spd = P('Vertical Speed', -x*np.sin(x) * 100)
        air_spd = P(
            name='Airspeed',
            array=np.ma.array(list(range(-2, 150, 5)) + list(range(150, -2, -5))),
        )

        air_spd.array[0] = 0
        descending=buildsections('Descending', [5,18],[39,57])
        node = self.node_class()
        node.derive(vrt_spd, air_spd, descending)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 12)
        self.assertAlmostEqual(node[0].value, -181, places=0)
        self.assertEqual(node[1].index, 49)
        self.assertAlmostEqual(node[1].value, -790, places=0)

    def test_derive_negative(self):
        vrt_spd = P('Vertical Speed', [1000.0]*15)
        air_spd = P('Airspeed', [70]*15)
        name = 'Descending'
        section = Section(name, slice(0, 16), 0, 16)
        descending = SectionNode(name, items=[section])
        node = self.node_class()
        node.derive(vrt_spd, air_spd, descending)
        self.assertEqual(len(node), 0)


class TestRateOfDescentAtHeightBeforeLevelOff(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescentAtHeightBeforeLevelFlight

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,
                         [('Vertical Speed',
                           'Altitude Before Level Flight When Descending')])

    def test_derive(self):
        roc_array = np.ma.concatenate((np.ones(19) * 250, [437, 625, 812, 1000, 1125,
                                                           625, 475, 500, 125, 375,
                                                           275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1-roc_array[::-1]))
        vert_spd = P('Vertical Speed', -roc_array)
        heights = AltitudeBeforeLevelFlightWhenDescending(
            'Altitude Before Level Flight When Descending',
            items=[
                KeyTimeInstance(22,
                                '2000 Ft Before Level Flight Descending'),
                KeyTimeInstance(24,
                                '1000 Ft Before Level Flight Descending'),
            ])
        node = self.node_class()
        node.derive(vert_spd, heights)

        expected = KPV('Rate Of Descent At Height Before Level Flight', items=[
            KeyPointValue(name='Rate Of Descent At 2000 Ft Before Level Off',
                          index=22,
                          value=-1000),
            KeyPointValue(name='Rate Of Descent At 1000 Ft Before Level Off',
                          index=24,
                          value=-625),
        ])
        self.assertEqual(node, expected)


##############################################################################
# Roll


class TestRollLiftoffTo20FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollLiftoffTo20FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Airborne')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (1, 20), {})]

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_basic(self):
        roll = P('Roll', np.ma.array([10.0]*5+[4.0]*10+[-9.0]*10))
        aal = P('Altitude AAL For Flight Phases', np.ma.arange(25))
        airs = buildsection('Airborne', 5, 30)
        kpv = self.node_class()
        kpv.derive(roll, aal, airs)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 15)
        self.assertEqual(kpv[0].value, -9.0)


class TestRoll20To400FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll20To400FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (20, 400), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll400To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Roll400To1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollAbove1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RollAbove1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1000,), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRoll1000To300FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Roll1000To300FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_abs_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll300To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll300To20FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (300, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll20FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Roll20FtToTouchdownMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Roll', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Roll', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Roll', 'Altitude AGL', 'Touchdown'), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesExceeding5DegDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesExceeding5DegDuringFinalApproach
        self.operational_combinations = [('Roll', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesExceeding15DegDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesExceeding15DegDuringFinalApproach
        self.operational_combinations = [('Roll', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesExceeding5DegDuringInitialClimb(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesExceeding5DegDuringInitialClimb
        self.operational_combinations = [('Roll', 'Initial Climb')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesExceeding15DegDuringInitialClimb(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesExceeding15DegDuringInitialClimb
        self.operational_combinations = [('Roll', 'Initial Climb')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesNotDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesNotDuringFinalApproach
        self.operational_combinations = [('Roll', 'Airborne', 'Final Approach', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

class TestRollAtLowAltitude(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollAtLowAltitude
        self.operational_combinations = [('Roll', 'Altitude Radio')]

    def test_basic(self):
        roll = P('Roll', np.ma.array([0.0]*2+[11.0]+[10.0]*5+[0.0]*2))
        alt_rad = P('Altitude Radio', np.ma.array([80.0]*10))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 2)
        self.assertEqual(kpv[0].value, 3)

    def test_negative_bank_angle(self):
        roll = P('Roll', np.ma.array([0.0]*2+[-11.0]+[10.0]*5+[0.0]*2))
        alt_rad = P('Altitude Radio', np.ma.array([80.0]*10))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(kpv[0].value, -3)

    def test_short_period(self):
        roll = P('Roll', np.ma.array([0.0]*2+[11.0]+[10.0]*4+[0.0]*3))
        alt_rad = P('Altitude Radio', np.ma.array([80.0]*10))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(len(kpv), 0)

    def test_high(self):
        roll = P('Roll', np.ma.array([0.0]*2+[60.0]*6+[0.0]*2))
        alt_rad = P('Altitude Radio', np.ma.array([650.0]*10))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(len(kpv), 0)

    def test_correct_peak(self):
        roll = P('Roll', np.ma.array([0.0]*2+[55]*3+[-26]*3+[0.0]*2))
        alt_rad = P('Altitude Radio', np.ma.array([500.0]*5+[200.0]*5))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 5)
        self.assertEqual(kpv[0].value, -6.0)

    def test_offset_index(self):
        roll = P('Roll', np.ma.array([0.0]*2+[55]*3+[-26]*13+[0.0]*2))
        alt_rad = P('Altitude Radio', np.ma.array([20]*4+[60.0]+[200.0]*15))
        kpv = self.node_class()
        kpv.derive(roll, alt_rad)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 4)
        self.assertEqual(kpv[0].value, 49.0)


class TestRollLeftBelow6000FtAltitudeDensityBelow60Kts(unittest.TestCase):

    def setUp(self):
        self.node_class = RollLeftBelow6000FtAltitudeDensityBelow60Kts

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter, family=A('Family', 'Puma'))
        self.assertEqual(opts, [('Roll', 'Altitude Density', 'Airspeed', 'Airborne')])

    def test_derive(self):
        alt = P('Altitude Density', np.ma.arange(0, 10000, 50))
        spd = P('Airspeed', np.ma.arange(0, 100, 0.5))
        x = np.linspace(0, 10, 200)
        roll = P('Roll', np.sin(x)*20)
        name = 'Airborne'
        section = Section(name, slice(10, 195), 10, 195)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, alt, spd, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 94)
        self.assertAlmostEqual(node[0].value, -19.999, places=3)

    def test_in_parts(self):
        # OK, too high, too fast, roll right, OK, not airborne
        alt = P('Altitude Density', np.ma.array([5000, 7000, 5000, 5000, 5000, 5000]))
        spd = P('Airspeed', np.ma.array([50, 50, 70, 50, 50, 50]))
        roll = P('Roll', np.ma.array([-66, -66, -66, +55, -66, -66]))
        airborne = buildsection('Airborne', 0, 5)
        node = self.node_class()
        node.derive(roll, alt, spd, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[1].index, 4)


class TestRollLeftBelow8000FtAltitudeDensityAbove60Kts(unittest.TestCase):

    def setUp(self):
        self.node_class = RollLeftBelow8000FtAltitudeDensityAbove60Kts

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter, family=A('Family', 'Puma'))
        self.assertEqual(opts, [('Roll', 'Altitude Density', 'Airspeed', 'Airborne')])

    def test_derive(self):
        alt = P('Altitude Density', np.ma.arange(0, 20000, 99))
        spd = P('Airspeed', np.ma.arange(0, 200))
        x = np.linspace(0, 10, 200)
        roll = P('Roll', np.sin(x)*20)
        name = 'Airborne'
        section = Section(name, slice(10, 195), 10, 195)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, alt, spd, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 80)
        self.assertAlmostEqual(node[0].value, -15.396, places=3)

    def test_in_parts(self):
        # OK, too high, too slow, roll right, OK, not airborne
        alt = P('Altitude Density', np.ma.array([7000, 9000, 7000, 7000, 7000, 7000]))
        spd = P('Airspeed', np.ma.array([70, 70, 50, 70, 70, 70, 70]))
        roll = P('Roll', np.ma.array([-66, -66, -66, +55, -66, -66]))
        airborne = buildsection('Airborne', 0, 5)
        node = self.node_class()
        node.derive(roll, alt, spd, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[1].index, 4)


class TestRollLeftAbove6000FtAltitudeDensityBelow60Kts(unittest.TestCase):

    def setUp(self):
        self.node_class = RollLeftAbove6000FtAltitudeDensityBelow60Kts

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter, family=A('Family', 'Puma'))
        self.assertEqual(opts, [('Roll', 'Altitude Density', 'Airspeed', 'Airborne')])

    def test_derive(self):
        alt = P('Altitude Density', np.ma.arange(0, 20000, 99))
        spd = P('Airspeed', np.ma.arange(0, 100, 0.5))
        x = np.linspace(0, 10, 200)
        roll = P('Roll', np.sin(x)*20)
        name = 'Airborne'
        section = Section(name, slice(10, 195), 10, 195)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, alt, spd, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 80)
        self.assertAlmostEqual(node[0].value, -15.396, places=3)

    def test_in_parts(self):
        # OK, too low, too FAST, roll right, OK, not airborne
        alt = P('Altitude Density', np.ma.array([7000, 5000, 7000, 7000, 7000, 7000]))
        spd = P('Airspeed', np.ma.array([50, 50, 70, 50, 50, 50]))
        roll = P('Roll', np.ma.array([-66, -66, -66, +55, -66, -66]))
        airborne = buildsection('Airborne', 0, 5)
        node = self.node_class()
        node.derive(roll, alt, spd, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[1].index, 4)



class TestRollLeftAbove8000FtAltitudeDensityAbove60Kts(unittest.TestCase):

    def setUp(self):
        self.node_class = RollLeftAbove8000FtAltitudeDensityAbove60Kts

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter, family=A('Family', 'Puma'))
        self.assertEqual(opts, [('Roll', 'Altitude Density', 'Airspeed', 'Airborne')])

    def test_derive(self):
        alt = P('Altitude Density', np.ma.arange(0, 20000, 100))
        spd = P('Airspeed', np.ma.arange(0, 200))
        x = np.linspace(0, 10, 200)
        roll = P('Roll', np.sin(x)*20)
        name = 'Airborne'
        section = Section(name, slice(10, 195), 10, 195)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, alt, spd, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 94)
        self.assertAlmostEqual(node[0].value, -19.999, places=3)

    def test_in_parts(self):
        # OK, too low, too SLOW, roll right, OK, not airborne
        alt = P('Altitude Density', np.ma.array([9000, 7000, 9000, 9000, 9000, 9000]))
        spd = P('Airspeed', np.ma.array([70, 70, 50, 70, 70, 70, 70]))
        roll = P('Roll', np.ma.array([-66, -66, -66, +55, -66, -66]))
        airborne = buildsection('Airborne', 0, 5)
        node = self.node_class()
        node.derive(roll, alt, spd, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[1].index, 4)


class TestRollAboveFL200Max(unittest.TestCase):
    def can_operate(self):
        opts = RollAboveFL200Max.get_operational_combinations()
        self.assertEqual(opts, [('Roll', 'Altitude STD Smoothed')])

    def test_derive(self):
        roll = P('Pitch', array=[4, 3, 1.5, 0, -2.5, 2, 1, -1, -3])
        alt = P('Altitude STD Smoothed',
                array=[1000, 2000, 7000, 10000, 40000, 22000, 11000, 9000, 8000])
        node = RollAboveFL200Max()
        node.derive(roll, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, -2.5)


class TestRollRateMaxAboveLimitAtTouchdown(unittest.TestCase):
    def test_high_roll_at_touchdown(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             2, 3, 4, 3, 2,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 4)


    def test_high_roll_before_td_window(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                                 0, 1, 2, 3, 4,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 0)


    def test_high_roll_beginning_of_window(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(12, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 2, 3, 4,
                                 5, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 5)


    def test_high_roll_end_of_window(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(8, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 2, 3, 4, 3, 2,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 2)


    def test_high_roll_after_landing(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 2, 3, 4, 5, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 0)


    def test_normal_landing(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns)

        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 0)


    def test_high_roll_bounce_after_normal_landing(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])
        bounces = buildsection('Bounced Landing', 13, 17)

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns, bounces=bounces)

        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 3)


    def test_normal_bounce_after_high_roll_landing(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])
        bounces = buildsection('Bounced Landing', 13, 17)

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             2, 3, 4, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns, bounces=bounces)

        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 4)


    def test_high_roll_bounce_after_high_roll_landing(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])
        bounces = buildsection('Bounced Landing', 13, 17)

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             2, 3, 4, 0, 0,
                             3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns, bounces=bounces)

        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[1].index, 16)
        self.assertEqual(node[1].value, 3)


    def test_touch_and_go_landing_and_bounce(self):
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(13, 'Touchdown')])
        touch_and_go = KTI('Touch And Go', items=[KeyTimeInstance(3, 'Touch And Go')])
        bounces = buildsection('Bounced Landing', 13, 17)

        limit=P('Roll Rate At Touchdown Limit',
                np.ma.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, ]))

        roll_rate=P('Roll Rate For Touchdown',
                    np.ma.array([0, 1, 3, 1, 0,
                             0, 0, 0, 0, 0,
                             2, 3, 4, 0, 0,
                             3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, ]))

        node = RollRateMaxAboveLimitAtTouchdown()
        node.derive(roll_rate, limit, touchdowns=touchdowns, touch_and_go=touch_and_go, bounces=bounces)

        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[1].index, 2)
        self.assertEqual(node[1].value, 3)
        self.assertEqual(node[2].index, 16)
        self.assertEqual(node[2].value, 3)


##############################################################################
# Rudder


class TestRudderDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RudderDuringTakeoffMax
        self.operational_combinations = [('Rudder',
                                          'Takeoff Roll Or Rejected Takeoff')]
        self.function = max_abs_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderCyclesAbove50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RudderCyclesAbove50Ft
        self.operational_combinations = [('Rudder', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderReversalAbove50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RudderReversalAbove50Ft
        self.operational_combinations = [('Rudder', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderPedalForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = RudderPedalForceMax
        self.operational_combinations = [('Rudder Pedal Force', 'Fast')]

    def test_derive(self):
        ccf = P(
            name='Rudder Pedal Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Rudder Pedal Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Rudder Pedal Force Max')]))

    def test_big_left_boot(self):
        ccf = P(
            name='Rudder Pedal Force',
            array=np.ma.array(data=range(30, -50, -5), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 13)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Rudder Pedal Force Max',
                items=[KeyPointValue(
                    index=12.0, value=-30.0,
                    name='Rudder Pedal Force Max')]))

    def test_derive_from_hdf(self):
        [rudder], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, '757-3A-001.hdf5'),
            ['Rudder Pedal Force'], slice(836, 21663), 'Fast')
        node = self.node_class()
        node.derive(rudder, phase)
        self.assertEqual(
            node,
            KPV('Rudder Pedal Force Max',
                items=[KeyPointValue(
                    index=21658.0, value=-23.944020961616012,
                    name='Rudder Pedal Force Max')]))


##############################################################################
# Speedbrake


class TestSpeedbrakeDeployed1000To20FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployed1000To20FtDuration
        self.operational_combinations = [('Speedbrake Selected', 'Altitude AAL For Flight Phases')]

    def test_derive_basic(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    array=np.ma.arange(2000, 0, -10))
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(
                [0] * 40 + [1] * 20 + [0] * 80 + [1] * 20 + [0] * 40))
        node = self.node_class()
        node.derive(spd_brk, alt_aal)
        self.assertEqual(
            node, [KeyPointValue(140, 20.0,
                                 'Speedbrake Deployed 1000 To 20 Ft Duration')])


class TestAltitudeWithSpeedbrakeDeployedDuringFinalApproachMin(
    unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeWithSpeedbrakeDeployedDuringFinalApproachMin
        self.vmap = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertIn('Altitude AAL', opts[0])
        self.assertIn('Speedbrake Selected', opts[0])
        self.assertIn('Final Approach', opts[0])

    def test_derive(self):
        alt = list(reversed(range(10, 1200,20))) + [1]*10
        brk = [0]*50 + [1]*11 + [0]*4 + [1]*5

        alt_aal = P('Altitude AAL', np.ma.array(alt))
        spd_brk = M('Speedbrake Selected', np.ma.array(brk),
                    values_mapping=self.vmap)
        fin_app = buildsection('Final Approach', 10, 58)

        node = self.node_class()
        node.derive(alt_aal=alt_aal, spd_brk=spd_brk, fin_app=fin_app)

        self.assertTrue(alt_aal.array.size == spd_brk.array.size)
        self.assertEqual(len(node),1)
        self.assertEqual(node[0].index, 57)
        self.assertEqual(node[0].value, 50)


class TestSpeedbrakeDeployedWithFlapDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithFlapDuration
        self.operational_combinations = [
            ('Speedbrake Selected', 'Flap Lever', 'Airborne'),
            ('Speedbrake Selected', 'Flap Lever (Synthetic)', 'Airborne'),
            ('Speedbrake Selected', 'Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]

    def test_derive_basic(self):
        array = np.ma.array(([0] * 4 + [1] * 2 + [0] * 4) * 3)
        mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M('Speedbrake Selected', array=array, values_mapping=mapping)
        airborne = buildsection('Airborne', 10, 20)
        name = self.node_class.get_name()

        array = np.ma.array([0] * 10 + list(range(1, 21)))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(spd_brk, flap_lever, None, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=14, value=2.0, name=name),
        ]))

        array = np.ma.array([1] * 10 + list(range(1, 21)))
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)
        node = self.node_class()
        node.derive(spd_brk, None, flap_synth, airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=14, value=2.0, name=name),
        ]))


class TestSpeedbrakeDeployedWithPowerOnDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithPowerOnDuration
        self.operational_combinations = [
            ('Speedbrake Selected', 'Eng (*) N1 Avg',
             'Altitude AAL For Flight Phases', 'Model')
        ]

    def test_derive_basic(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        power = P('Eng (*) N1 Avg',
                  array=np.ma.concatenate((np.ones(10) * 40, np.ones(10) * 60, np.ones(10) * 50)))
        aal = P('Altitude AAL For Flight Phases',
                array=np.ma.concatenate((np.arange(0, 75, 5), np.arange(70, -5, -5))))
        node = self.node_class()
        node.derive(spd_brk, power, aal, None)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0,
                          'Speedbrake Deployed With Power On Duration')])

    def test_derive_CRJ200ER(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        power = P('Eng (*) N1 Avg',
                  array=np.ma.concatenate((np.ones(10) * 40, np.ones(10) * 80, np.ones(10) * 70)))
        aal = P('Altitude AAL For Flight Phases',
                array=np.ma.concatenate((np.arange(0, 150, 5), np.arange(150, -5, -5))))
        node = self.node_class()
        node.derive(spd_brk, power, aal, A('Model', 'CRJ200ER (CL-600-2B19)'))
        self.assertEqual(node, [
            KeyPointValue(14, 2.0,
                          'Speedbrake Deployed With Power On Duration')])


class TestSpeedbrakeDeployedDuringGoAroundDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedDuringGoAroundDuration
        self.operational_combinations = [('Speedbrake Selected', 'Go Around And Climbout')]

    def test_derive(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        go_around = buildsection('Go Around And Climbout', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, go_around)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0,
                          'Speedbrake Deployed During Go Around Duration')])


##############################################################################
# Warnings: Stick Pusher/Shaker

class TestStallWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Stall Warning'
        self.phase_name = 'Airborne'
        self.node_class = StallWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}
        self.basic_setup()


class TestStickPusherActivatedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Stick Pusher'
        self.phase_name = 'Airborne'
        self.node_class = StickPusherActivatedDuration
        self.values_mapping = {0: '-', 1: 'Push'}
        self.basic_setup()


class TestStickShakerActivatedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Stick Shaker'
        self.phase_name = 'Airborne'
        self.node_class = StickShakerActivatedDuration
        self.values_mapping = {0: '-', 1: 'Shake'}
        self.basic_setup()

    def test_short_pulses(self):
        # We have seen aircraft with one sensor stuck on. This checks that
        # the KPV does not record in this condition.
        shaker = [0, 1]
        values_mapping = {0: '-', 1: 'Shake'}
        shaker = M(
            'Stick Shaker', values_mapping=values_mapping,
            array=np.ma.array(shaker * 5), frequency=2.0)
        airs = buildsection('Airborne', 0, 20)
        node = StickShakerActivatedDuration()
        node.derive(shaker, airs)
        self.assertEqual(node, [])

class TestOverspeedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Overspeed Warning'
        self.phase_name = 'Airborne'
        self.node_class = OverspeedDuration
        self.values_mapping = {0: '-', 1: 'Overspeed'}
        self.basic_setup()


class TestStallFaultCautionDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = StallFaultCautionDuration
        self.vmap = {0:"-", 1:"Caution"}

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.SECOND)
        self.assertEqual(node.name, 'Stall Fault Caution Duration')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts),3)
        for opt in opts:
            self.assertIn('Airborne', opt)
            stall_l = 'Stall (L) Fault Caution' in opt
            stall_r = 'Stall (R) Fault Caution' in opt
            self.assertTrue(stall_l or stall_r)

    def test_derive(self):
        l = np.ma.array([0]*10 + [1]*6 + [0]*14 + [1]*5 + [0]*15)
        r = np.ma.array([0]*14 + [1]*6 + [0]*20 + [1]*5 + [0]*5)
        self.assertEqual(len(l),len(r))
        self.assertEqual(len(l),50)

        stall_l = M('Stall (L) Fault Caution', l, values_mapping=self.vmap)
        stall_r = M('Stall (R) Fault Caution', r, values_mapping=self.vmap)

        airborne = buildsection('Airborne', 1, 48)

        node = self.node_class()
        node.derive(stall_l, stall_r, airborne)

        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 5)
        self.assertEqual(node[2].index, 40)
        self.assertEqual(node[2].value, 5)

    def test_stall_l(self):
        l = np.ma.array([0]*10 + [1]*6 + [0]*14 + [1]*5 + [0]*15)
        self.assertEqual(len(l),50)

        stall_l = M('Stall (L) Fault Caution', l, values_mapping=self.vmap)
        stall_r = None

        airborne = buildsection('Airborne', 1, 48)

        node = self.node_class()
        node.derive(stall_l, stall_r, airborne)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 5)


class TestCruiseSpeedLowDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Cruise Speed Low'
        self.phase_name = 'Airborne'
        self.node_class = CruiseSpeedLowDuration
        self.values_mapping = {0: '-', 1: 'Low'}
        self.basic_setup()


class TestDegradedPerformanceCautionDuration(unittest.TestCase,
                                             CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Degraded Performance Caution'
        self.phase_name = 'Airborne'
        self.node_class = DegradedPerformanceCautionDuration
        self.values_mapping = {0: '-', 1: 'Caution'}
        self.basic_setup()


class TestAirspeedIncreaseAlertDuration(unittest.TestCase,
                                        CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Airspeed Increase Alert'
        self.phase_name = 'Airborne'
        self.node_class = AirspeedIncreaseAlertDuration
        self.values_mapping = {0: '-', 1: 'Alert'}
        self.basic_setup()


class TestAirspeedBelowMinimumAirspeedFlapCleanMin(unittest.TestCase):
    def setUp(self):
        self.node_class = AirspeedBelowMinimumAirspeedFlapCleanMin
        self.air_spd = P('Airspeed',
                         np.ma.array([
                             170, 170, 171, 172, 172, 172, 174, 174, 175, 176,
                             176, 176, 177, 177, 178, 178, 178, 178, 178, 178,
                             178, 178, 177, 176, 175, 174, 172, 172, 172, 168,
                             168, 168, 166, 166, 165, 165, 166, 167, 169, 171,
                             174, 178, 182, 184, 188, 190, 194, 196, 198, 200,
                             202, 206, 208, 211, 213, 214, 216, 217, 218, 220,
                             220, 221, 221, 222, 222, 224, 224
                         ]))
        self.f_m_spd = P('Flap Manoeuvre Speed',
                         np.ma.array([
                             156, 156, 156, 156, 156, 156, 176, 176, 176, 176,
                             176, 176, 176, 176, 176, 176, 176, 176, 176, 176,
                             176, 176, 176, 176, 176, 176, 196, 196, 196, 196,
                             196, 196, 196, 196, 196, 196, 196, 196, 196, 196,
                             196, 196, 196, 196, 196, 196, 196, 196, 196, 196,
                             196, 196, 196, 196, 196, 196, 196, 196, 196, 196,
                             196, 196, 196, 196, 196, 196, 196
                             ], mask=[False]*40 + [True]*27))
        self.min_spd = P('Minimum Airspeed',
                         np.ma.array([
                             156, 156, 156, 156, 156, 156, 176, 176, 176, 176,
                             176, 176, 176, 176, 176, 176, 176, 176, 176, 176,
                             176, 176, 176, 176, 176, 176, 190, 190, 190, 190,
                             190, 190, 190, 190, 190, 190, 190, 190, 190, 190,
                             190, 190, 190, 190, 190, 190, 190, 190, 190, 190,
                             190, 190, 190, 190, 190, 190, 190, 190, 190, 190,
                             190, 190, 190, 190, 190, 190, 190
                         ]))
        self.flap = M('Flap Including Transition',
                      np.ma.array([
                          5, 5, 5, 5, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0
                          ]),
                      values_mapping={0:"0", 1: "1", 5: "5", 15: "15",
                                      20: "20", 25: "25", 30: "30"})
        self.airborne = buildsection('Airborne', 0, 66)

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.SECOND)
        self.assertEqual(node.name, 'Airspeed Below Minimum Airspeed Flap Clean Min')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)
        for opt in opts:
            self.assertIn('Airspeed', opt)
            self.assertIn('Airborne', opt)
            self.assertIn('Flap Including Transition', opt)
            self.assertTrue(any_of(['Flap Manoeuvre Speed','Minimum Airspeed',],
                                   opt))

    def test_derive_flap_manoeuvre_speed(self):
        '''
        If Minimum Airspeed isn't available use Flap Manoeuvre Speed
        Data from the example flight in the ticket. Duration: 19, Index 28
        '''
        node = self.node_class()
        node.derive(air_spd=self.air_spd,
                    min_spd=None,
                    flap=self.flap,
                    f_m_spd=self.f_m_spd,
                    airborne=self.airborne,)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 34)
        self.assertEqual(node[0].value, -31)

    def test_derive_minimum_airspeed(self):
        '''
        Use just the Minimum Airspeed parameter. The test should
        be 2 less than the kpv from test_derive_flap_manoeuvre_speed
        Duration: 17, Index 28
        '''
        node = self.node_class()
        node.derive(air_spd=self.air_spd,
                    min_spd=self.min_spd,
                    flap=self.flap,
                    f_m_spd=None,
                    airborne=self.airborne,)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 34)
        self.assertEqual(node[0].value, -25)

    def test_derive_both_speed_param(self):
        '''
        Uses Minimum Airspeed parameter and Flap Manoeuvre Speed. The KPV
        should use Minimum Speed over Flap Manoeuvre Speed. The test should
        be 2 less than the kpv from test_derive_flap_manoeuvre_speed
        Duration: 17, Index 28
        '''
        node = self.node_class()
        node.derive(air_spd=self.air_spd,
                    min_spd=self.min_spd,
                    flap=self.flap,
                    f_m_spd=self.f_m_spd,
                    airborne=self.airborne,)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 34)
        self.assertEqual(node[0].value, -25)


##############################################################################
# Tail Clearance


class TestTailClearanceDuringTakeoffMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TailClearanceDuringTakeoffMin
        self.operational_combinations = [('Altitude Tail', 'Takeoff')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceDuringLandingMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TailClearanceDuringLandingMin
        self.operational_combinations = [('Altitude Tail', 'Landing')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceDuringApproachMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TailClearanceDuringApproachMin
        self.operational_combinations = [('Altitude AAL', 'Altitude Tail', 'Distance To Landing')]

    @unittest.skip('Test Out Of Date')
    def test_derive(self):
        # XXX: The BDUTerrain test files are missing from the repository?
        test_data_dir = os.path.join(test_data_path, 'BDUTerrain')
        alt_aal_array = np.ma.masked_array(load_compressed(os.path.join(test_data_dir, 'alt_aal.npz')))
        alt_radio_array = np.ma.masked_array(load_compressed(os.path.join(test_data_dir, 'alt_radio.npz')))
        dtl_array = np.ma.masked_array(load_compressed(os.path.join(test_data_dir, 'dtl.npz')))
        alt_aal = P(array=alt_aal_array, frequency=8)
        alt_radio = P(array=alt_radio_array, frequency=0.5)
        dtl = P(array=dtl_array, frequency=0.25)
        alt_radio.array = align(alt_radio, alt_aal)
        dtl.array = align(dtl, alt_aal)
        # FIXME: Should tests for the BDUTerrain node be in a separate TestCase?
        ####param = BDUTerrain()
        ####param.derive(alt_aal, alt_radio, dtl)
        ####self.assertEqual(param, [
        ####    KeyPointValue(name='BDU Terrain', index=1008, value=0.037668517049960347),
        ####])


##############################################################################
# Temperature


class TestSATMax(unittest.TestCase):

    def setUp(self):
        self.node_class = SATMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.CELSIUS)
        self.assertEqual(node.name, 'SAT Max')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('SAT',)])

    def test_derive(self,):
        sat = P('SAT', np.ma.arange(0, 11))
        node = self.node_class()
        node.derive(sat)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)


##############################################################################
# Terrain Clearance


class TestTerrainClearanceAbove3000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TerrainClearanceAbove3000FtMin
        self.operational_combinations = [('Altitude Radio', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_above', (3000.0,), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Tailwind


# FIXME: Make CreateKPVsWithinSlicesTest more generic and then use it again...
class TestTailwindLiftoffTo100FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TailwindLiftoffTo100FtMax
        self.operational_combinations = [('Tailwind', 'Altitude AAL For Flight Phases')]
        #self.second_param_method_calls = [('slices_from_to', (0, 100), {})]
        #self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# FIXME: Make CreateKPVsWithinSlicesTest more generic and then use it again...
class TestTailwind100FtToTouchdownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Tailwind100FtToTouchdownMax
        self.operational_combinations = [('Tailwind', 'Altitude AAL For Flight Phases', 'Approach And Landing')]

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Tailwind', 'Altitude AAL For Flight Phases', 'Approach And Landing')])

    def test_derive(self):
        x = np.linspace(0,10,200)
        tailwind = P('Tailwind', x*np.sin(x)*3)
        landing = buildsection('Approach And Landing', 10, 150)
        alt_aal = P('Altitude AAL For Flight Phases', np.linspace(150,20,200))
        node = self.node_class()
        node.derive(tailwind, alt_aal, landing)

        self.assertEqual(len(node), 1)


class TestTailwindDuringTakeoffMax(unittest.TestCase):

    def setUp(self):
        self.node_class = TailwindDuringTakeoffMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(opts, [('Tailwind', 'Airspeed True', 'Takeoff', 'Takeoff Rotation')])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [])

    def test_derive(self,):
        x = np.linspace(0, 10, 200)
        tailwind = P('Tailwind', x*np.sin(x)*3)
        ias = P('Airspeed', np.ma.concatenate((np.zeros(125), np.arange(10,160,2))))
        ias.array[:126] = np.ma.masked
        toff = buildsection('Takeoff', 50, 185)
        toff_rotation = buildsection('Takeoff Rotation', 165, 190)

        node = self.node_class()
        node.derive(tailwind, ias, toff, toff_rotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 159)
        self.assertAlmostEqual(node[0].value, 23.75, places=2)

    def test_derive__masked_below_100(self,):
        x = np.linspace(0, 10, 200)
        tailwind = P('Tailwind', x*np.sin(x)*3)
        spd_array = np.ma.concatenate((np.zeros(125), np.arange(10,160,2)))
        spd_array = np.ma.masked_less_equal(spd_array, 100)
        ias = P('Airspeed', spd_array)
        toff = buildsection('Takeoff', 50, 185)
        toff_rotation = buildsection('Takeoff Rotation', 175, 190)

        node = self.node_class()
        node.derive(tailwind, ias, toff, toff_rotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 171)
        self.assertAlmostEqual(node[0].value, 19.05, places=2)

    def test_derive__tailwind_high_during_rotation(self):
        # Name not really true
        x = np.linspace(0, 10, 200)
        spd_array = np.ma.concatenate(([0]*125, np.ma.arange(10,160,2)))
        spd_array = np.ma.masked_less_equal(spd_array, 100)

        tailwind = P('Tailwind', x*np.sin(x)*3)
        ias = P('Airspeed', spd_array)
        toff = buildsection('Takeoff', 50, 185)
        toff_rotation = buildsection('Takeoff Rotation', 158, 190)

        node = self.node_class()
        node.derive(tailwind, ias, toff, toff_rotation)

        self.assertEqual(len(node), 0)


class TestFuelQtyLowWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQtyLowWarningDuration.get_operational_combinations()
        self.assertEqual(opts, [('Fuel Qty (*) Low', 'Airborne')])

    def test_derive(self):
        low = FuelQtyLowWarningDuration()
        warn = M(array=np.ma.array([0,0,1,1,1]),
                 values_mapping={1: 'Warning'})
        air = buildsection('Airborne', 1, 3)
        low.derive(warn, air)
        self.assertEqual(low[0].index, 2)
        self.assertEqual(low[0].value, 2)


##############################################################################
# Warnings: Takeoff Configuration Warning


class TestMasterWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarningDuration
        self.operational_combinations = [('Master Warning','Eng (*) Any Running'),
                                         ('Master Warning',)]

    def test_derive(self):
        warn = MasterWarningDuration()
        warn.derive(M(array=np.ma.array([0,1,1,1,0]),
                      values_mapping={1: 'Warning'}),
                    M(array=np.ma.array([0,0,1,1,0]),
                     values_mapping={1: 'Running'})                   )
        self.assertEqual(warn[0].index, 2)
        self.assertEqual(warn[0].value, 2)

    def test_no_engines(self):
        warn = MasterWarningDuration()
        warn.derive(M(array=np.ma.array([0,1,1,1,0]),
                      values_mapping={1: 'Warning'}),None)
        self.assertEqual(warn[0].index, 1)
        self.assertEqual(warn[0].value, 3)

    def test_derive_all_running(self):
        warn = MasterWarningDuration()
        warn.derive(M(array=np.ma.array([0,1,1,1,1]),
                      values_mapping={1: 'Warning'}),
                    M(array=np.ma.array([1,1,1,1,1]),
                     values_mapping={1: 'Running'})                   )
        self.assertEqual(warn[0].index, 1)
        self.assertEqual(warn[0].value, 4)

    def test_derive_not_running(self):
        warn = MasterWarningDuration()
        warn.derive(M(array=np.ma.array([0,1,1,1,0]),
                      values_mapping={1: 'Warning'}),
                    M(array=np.ma.array([0,0,0,0,0]),
                     values_mapping={1: 'Running'})                   )
        self.assertEqual(len(warn), 0)

    def test_derive_AW139(self):
        master_warning = M(array=np.ma.array([0,0,0,1,1,1,1,1,1,1,0,0]),
                           values_mapping={1: 'Warning'})
        family = A('Family', value='AW139')
        phase = buildsection('Airborne', 5, 7)
        node = self.node_class()
        node.derive(master_warning, None, family, phase)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 3)

    def test_derive_single_sample(self):
        node = MasterWarningDuration()
        master_warning = M(array=np.ma.array([0,1]*10),
                           values_mapping={1: 'Warning'},
                           )
        engine_running = M(array=np.ma.array([1]*20),
                           values_mapping={1:'Running'},
                           )
        node.derive(master_warning, engine_running)
        self.assertEqual(len(node), 0)


class TestEngRunningDuration(unittest.TestCase):

    def test_can_operate(self):
        opts = EngRunningDuration.get_operational_combinations()
        self.assertEqual(len(opts), 15)
        self.assertIn(('Eng (1) Running',), opts)
        self.assertIn(('Eng (2) Running',),  opts)
        self.assertIn(('Eng (3) Running',), opts)
        self.assertIn(('Eng (4) Running',),  opts)
        self.assertIn(('Eng (1) Running', 'Eng (2) Running'), opts)
        self.assertIn(('Eng (1) Running', 'Eng (2) Running', 'Eng (3) Running'), opts)
        self.assertIn(('Eng (1) Running', 'Eng (2) Running', 'Eng (3) Running', 'Eng (4) Running'), opts)

    def test_derive(self):
        eng1 = M('Eng (1) Running', np.ma.array([0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0]),
                 values_mapping={0: 'Not Running', 1: 'Running'})
        eng2 = M('Eng (2) Running', np.ma.array([0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,1,0,0]),
                 values_mapping={0: 'Not Running', 1: 'Running'})
        eng3 = M('Eng (3) Running', np.ma.array([0,1,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0]),
                 values_mapping={0: 'Not Running', 1: 'Running'})
        eng4 = None

        running = EngRunningDuration()
        running.derive(eng1, eng2, eng3, eng4)

        self.assertEqual(len(running), 3)
        self.assertEqual(running[0].name, 'Eng (1) Running Duration')
        self.assertEqual(running[0].index, 10)
        self.assertEqual(running[0].value, 6)
        self.assertEqual(running[1].name, 'Eng (2) Running Duration')
        self.assertEqual(running[1].index, 11)
        self.assertEqual(running[1].value, 5)
        self.assertEqual(running[2].name, 'Eng (3) Running Duration')
        self.assertEqual(running[2].index, 11)
        self.assertEqual(running[2].value, 6)


class TestEngStartTimeMax(unittest.TestCase):

    def test_can_operate(self):
        ac_type=A('Aircraft Type', value='aeroplane')
        ac_family=A('Family', value='Phenom 300')
        opts = EngStartTimeMax.get_operational_combinations(ac_type=ac_type,
                                                            family=ac_family)
        self.assertEqual(opts,  [('Eng (1) N2', 'Eng (2) N2',
                                  'Stationary', 'Taxi Out')])

    def test_1_start(self):
        eng_1_start = P('Eng (1) N2', np.ma.array([0]*4+list(range(55))+[54]*4))
        eng_2_start = P('Eng (2) N2', np.ma.array([0]*50))
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_1_start, eng_2_start, stat, t_out)
        self.assertEqual(len(estm), 1)
        self.assertEqual(estm[0].value, 53.0)
        self.assertEqual(estm[0].index, 5.0)

    def test_no_start(self):
        eng_start = P('Eng (1) N2', np.ma.array([0]*50))
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_start, eng_start, stat, t_out)
        self.assertEqual(len(estm), 0)

    def test_early_cut(self):
        eng_start = P('Eng (1) N2',
                      np.ma.array([60, 30]+[0]*5+list(range(0, 60, 5))+[60]*3),
                      frequency=0.5)
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_start, eng_start, stat, t_out)
        self.assertEqual(len(estm), 1)
        self.assertEqual(estm[0].value, 22.0)
        self.assertEqual(estm[0].index, 8.0)

    def test_unclear_start(self):
        eng_start = P('Eng (1) N2', np.ma.array([0]*5+list(range(45))+[45]*5))
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_start, eng_start, stat, t_out)
        self.assertEqual(len(estm), 0)

    def test_2_start(self):
        eng_1_start = P('Eng (1) N2', np.ma.array([60.0]*50))
        eng_2_start = P('Eng (2) N2', np.ma.array([0]*3+list(range(0, 55, 2))+[54]*4))
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_1_start, eng_2_start, stat, t_out)
        self.assertEqual(len(estm), 1)
        self.assertEqual(estm[0].value, 26.0)
        self.assertEqual(estm[0].index, 4.0)

    def test_both_start(self):
        eng_1_start = P('Eng (1) N2', np.ma.array([0]*4+list(range(55))+[54]*4))
        eng_2_start = P('Eng (2) N2', np.ma.array([0]*6+list(range(55))+[54]*2))
        stat=buildsection('Stationary', 0, 25)
        t_out=buildsection('Taxi Out', 20, 70)
        estm = EngStartTimeMax()
        estm.derive(eng_1_start, eng_2_start, stat, t_out)
        self.assertEqual(len(estm), 1)
        self.assertEqual(estm[0].value, 53.0)


class TestMasterWarningDuringTakeoffDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarningDuringTakeoffDuration
        self.operational_combinations = [('Master Warning',
                                          'Takeoff Roll Or Rejected Takeoff')]

    def test_derive(self):
        node = self.node_class()
        takeoff_rolls = buildsection('Takeoff Roll Or Rejected Takeoff', 5, 15)
        master_warning = M(array=np.ma.array([0, 1, 1, 1, 0,
                                              0, 0, 0, 0, 0,
                                              0, 1, 1, 1, 0,
                                              0, 0, 0, 0, 0,
                                              0, 1, 1, 1, 0,
                                              ]),
                           values_mapping={1: 'Warning'},
                           )
        node.derive(master_warning, takeoff_rolls)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 3)

    def test_derive_single_sample(self):
        node = self.node_class()
        takeoff_rolls = buildsection('Takeoff Roll Or Rejected Takeoff', 0, 20)
        master_warning = M(array=np.ma.array([0,1]*10),
                           values_mapping={1: 'Warning'},
                           )
        node.derive(master_warning, takeoff_rolls)
        self.assertEqual(len(node), 0)


class TestMasterCautionDuringTakeoffDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterCautionDuringTakeoffDuration
        self.operational_combinations = [('Master Caution',
                                          'Takeoff Roll Or Rejected Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')




##############################################################################
# Warnings: Landing Configuration Warning


class TestLandingConfigurationGearWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = LandingConfigurationGearWarningDuration.get_operational_combinations()
        self.assertEqual(opts, [('Landing Configuration Gear Warning', 'Airborne',)])

    def test_derive(self):
        node = LandingConfigurationGearWarningDuration()
        airs = buildsection('Airborne', 2, 8)
        warn = M(array=np.ma.array([0,0,0,0,0,1,1,0,0,0]),
                 values_mapping={1: 'Warning'})
        node.derive(warn, airs)
        self.assertEqual(node[0].index, 5)


class TestLandingConfigurationSpeedbrakeCautionDuration(unittest.TestCase,
                                                        CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'Landing Configuration Speedbrake Caution'
        self.phase_name = 'Airborne'
        self.node_class = LandingConfigurationSpeedbrakeCautionDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


##############################################################################
# Taxi In


class TestTaxiInDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = TaxiInDuration.get_operational_combinations()
        self.assertEqual(opts, [('Taxi In',)])

    def test_derive(self):
        taxi_ins = buildsections('Taxi In', [5, 9], [20, 29])
        node = TaxiInDuration()
        node.derive(taxi_ins)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0], KeyPointValue(7.5, 5, 'Taxi In Duration'))
        self.assertEqual(node[1], KeyPointValue(25, 10, 'Taxi In Duration'))


##############################################################################
# Taxi Out


class TestTaxiOutDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = TaxiOutDuration.get_operational_combinations()
        self.assertEqual(opts, [('Taxi Out',)])

    def test_derive(self):
        taxi_outs = buildsections('Taxi Out', [35, 66])
        node = TaxiOutDuration()
        node.derive(taxi_outs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0], KeyPointValue(51, 32, 'Taxi Out Duration'))


##############################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TestTAWSAlertDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSAlertDuration
        self.operational_combinations = [('TAWS Alert', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Warning'
        self.phase_name = 'Airborne'
        self.node_class = TAWSWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSGeneralWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGeneralWarningDuration
        self.operational_combinations = [('TAWS General', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSSinkRateWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSSinkRateWarningDuration
        self.operational_combinations = [('TAWS Sink Rate', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowFlapWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowFlapWarningDuration
        self.operational_combinations = [('TAWS Too Low Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTerrainWarningDuration
        self.operational_combinations = [('TAWS Terrain', 'Airborne'),
                                         ('TAWS Terrain Warning', 'Airborne'),
                                         ('TAWS Terrain', 'TAWS Terrain Warning', 'Airborne')]

    def test_derive_basic(self):
        values_mapping = {1: 'Warning', 0: '-'}
        taws_terrain_array = np.ma.array([0] * 10 + [1] * 10 + [0] * 20)
        taws_terrain_warning_array = np.ma.array(
            [0] * 15 + [1] * 10 + [0] * 5 + [1] * 5 + [0] * 5)
        taws_terrain = M('TAWS Terrain', array=taws_terrain_array,
                         values_mapping=values_mapping)
        taws_terrain_warning = M('TAWS Terrain Warning',
                                 array=taws_terrain_warning_array,
                                 values_mapping=values_mapping)
        airborne = buildsection('Airborne', 12, 33)
        node = self.node_class()
        node.derive(taws_terrain, None, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 8)
        node = self.node_class()
        node.derive(None, taws_terrain_warning, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 4)
        node = self.node_class()
        node.derive(taws_terrain, taws_terrain_warning, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 12)
        self.assertEqual(node[0].value, 13)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 4)


class TestTAWSTerrainPullUpWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTerrainPullUpWarningDuration
        self.operational_combinations = [('TAWS Terrain Pull Up', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainClearanceFloorAlertDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = TAWSTerrainClearanceFloorAlertDuration
        self.operational_combinations = [('TAWS Terrain Clearance Floor Alert',
                                          'Airborne')]

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)
        for opt in opts:
            self.assertIn('Airborne', opt)
            alert1 = 'TAWS Terrain Clearance Floor Alert' in opt
            alert2 = 'TAWS Terrain Clearance Floor Alert (2)' in opt
            self.assertTrue(alert1 or alert2)

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.SECOND)
        self.assertEqual(node.name,
                         'TAWS Terrain Clearance Floor Alert Duration')

    def test_derive(self):
        alerts1 = M('TAWS Terrain Clearance Floor Alert',
                    np.ma.array([0]*15 + [1]*15 + [0]*10),
                    values_mapping={0:'-', 1:'Alert'})
        alerts2 = None
        phase = buildsection('Airborne', 0, 24)

        node = self.node_class()
        node.derive(alerts1, alerts2, phase)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 10)

    def test_derive_2(self):
        alerts1 = None
        alerts2 = M('TAWS Terrain Clearance Floor Alert (2)',
                    np.ma.array([0]*15 + [1]*15 + [0]*10),
                    values_mapping={0:'-', 1:'Alert'})
        phase = buildsection('Airborne', 0, 24)

        node = self.node_class()
        node.derive(alerts1, alerts2, phase)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 10)

    def test_derive_3(self):
        alerts1 = M('TAWS Terrain Clearance Floor Alert',
                    np.ma.array([0]*15 + [1]*5 + [0]*20),
                    values_mapping={0:'-', 1:'Alert'})
        alerts2 = M('TAWS Terrain Clearance Floor Alert (2)',
                    np.ma.array([0]*20 + [1]*10 + [0]*10),
                    values_mapping={0:'-', 1:'Alert'})
        phase = buildsection('Airborne', 0, 24)

        node = self.node_class()
        node.derive(alerts1, alerts2, phase)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 10)

    def test_derive_4(self):
        alerts1 = M('TAWS Terrain Clearance Floor Alert',
                    np.ma.array([0]*10 + [1]*5 + [0]*25),
                    values_mapping={0:'-', 1:'Alert'})
        alerts2 = M('TAWS Terrain Clearance Floor Alert (2)',
                    np.ma.array([0]*25 + [1]*10 + [0]*5),
                    values_mapping={0:'-', 1:'Alert'})
        phase = buildsection('Airborne', 0, 29)

        node = self.node_class()
        node.derive(alerts1, alerts2, phase)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[1].index, 25)
        self.assertEqual(node[1].value, 5)


class TestTAWSGlideslopeWarning1500To1000FtDuration(unittest.TestCase,
                                                    NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning1500To1000FtDuration
        self.alt_aal = P('Altitude AAL For Flight Phases',
                         np.linspace(1700.0, 900.0, 50))
        self.warn_map = {0: '-', 1: 'Warning'}
        self.alert_map = {0: '-', 1: 'Warning'}

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'TAWS Glideslope Warning 1500 To 1000 Ft Duration')
        self.assertEqual(node.units, ut.SECOND)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)

        for opt in opts:
            self.assertIn('Altitude AAL For Flight Phases', opt)
            warn = 'TAWS Glideslope' in opt
            alert = 'TAWS Glideslope Alert' in opt
            self.assertTrue(warn or alert)

    def test_derive(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(10), np.ones(15), np.zeros(25))),
                            values_mapping=self.warn_map)

        node = self.node_class()
        node.derive(taws_glideslope, None, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 12)

    def test_derive_2(self):
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(None, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 25)
        self.assertEqual(node[0].value, 10)

    def test_derive_3(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(15), np.ones(15), np.zeros(20))),
                            values_mapping=self.warn_map)
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 20)

    def test_derive_4(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(23), np.ones(5), np.zeros(22))),
                            values_mapping=self.warn_map)

        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(30), np.ones(10), np.zeros(10))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 23)
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 10)


class TestTAWSGlideslopeWarning1000To500FtDuration(unittest.TestCase,
                                                   NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning1000To500FtDuration
        self.alt_aal = P('Altitude AAL For Flight Phases',
                         np.linspace(1200.0, 400.0, 50))
        self.warn_map = {0: '-', 1: 'Warning'}
        self.alert_map = {0: '-', 1: 'Warning'}

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'TAWS Glideslope Warning 1000 To 500 Ft Duration')
        self.assertEqual(node.units, ut.SECOND)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)

        for opt in opts:
            self.assertIn('Altitude AAL For Flight Phases', opt)
            warn = 'TAWS Glideslope' in opt
            alert = 'TAWS Glideslope Alert' in opt
            self.assertTrue(warn or alert)

    def test_derive(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(10), np.ones(15), np.zeros(25))),
                            values_mapping=self.warn_map)

        node = self.node_class()
        node.derive(taws_glideslope, None, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 12)

    def test_derive_2(self):
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(None, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 25)
        self.assertEqual(node[0].value, 10)

    def test_derive_3(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(15), np.ones(15), np.zeros(20))),
                            values_mapping=self.warn_map)
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(node[0].value, 20)

    def test_derive_4(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(23), np.ones(5), np.zeros(22))),
                            values_mapping=self.warn_map)
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(30), np.ones(10), np.zeros(10))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 23)
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 10)


class TestTAWSGlideslopeWarning500To200FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning500To200FtDuration
        self.alt_aal = P('Altitude AAL For Flight Phases',
                         np.linspace(800.0, 100.0, 50))
        self.warn_map = {0: '-', 1: 'Warning'}
        self.alert_map = {0: '-', 1: 'Warning'}

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'TAWS Glideslope Warning 500 To 200 Ft Duration')
        self.assertEqual(node.units, ut.SECOND)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 3)

        for opt in opts:
            self.assertIn('Altitude AAL For Flight Phases', opt)
            warn = 'TAWS Glideslope' in opt
            alert = 'TAWS Glideslope Alert' in opt
            self.assertTrue(warn or alert)

    def test_derive(self):
        taws_glideslope = M('TAWS Glideslope',
                            np.ma.concatenate((np.zeros(15), np.ones(10), np.zeros(25))),
                            values_mapping=self.warn_map)

        node = self.node_class()
        node.derive(taws_glideslope, None, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 22)
        self.assertEqual(node[0].value, 3)

    def test_derive_2(self):
        taws_alert = M('TAWS Glideslope Alert',
                       np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                       values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(None, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 25)
        self.assertEqual(node[0].value, 10)

    def test_derive_3(self):
        taws_glideslope = M('TAWS Glideslope',
                                np.ma.concatenate((np.zeros(15), np.ones(10), np.zeros(25))),
                                values_mapping=self.warn_map)
        taws_alert = M('TAWS Glideslope Alert',
                           np.ma.concatenate((np.zeros(25), np.ones(10), np.zeros(15))),
                           values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 22)
        self.assertEqual(node[0].value, 13)

    def test_derive_4(self):
        taws_glideslope = M('TAWS Glideslope',
                                np.ma.concatenate((np.zeros(23), np.ones(5), np.zeros(22))),
                                values_mapping=self.warn_map)
        taws_alert = M('TAWS Glideslope Alert',
                           np.ma.concatenate((np.zeros(30), np.ones(10), np.zeros(10))),
                           values_mapping=self.alert_map)

        node = self.node_class()
        node.derive(taws_glideslope, taws_alert, self.alt_aal)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 23)
        self.assertEqual(node[0].value, 5)
        self.assertEqual(node[1].index, 30)
        self.assertEqual(node[1].value, 10)


class TestTAWSTooLowTerrainWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowTerrainWarningDuration
        self.operational_combinations = [('TAWS Too Low Terrain', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowGearWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowGearWarningDuration
        self.operational_combinations = [('TAWS Too Low Gear', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSPullUpWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSPullUpWarningDuration
        self.operational_combinations = [('TAWS Pull Up', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSDontSinkWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Dont Sink'
        self.phase_name = 'Airborne'
        self.node_class = TAWSDontSinkWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSCautionObstacleDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Caution Obstacle'
        self.phase_name = 'Airborne'
        self.node_class = TAWSCautionObstacleDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSCautionTerrainDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Caution Terrain'
        self.phase_name = 'Airborne'
        self.node_class = TAWSCautionTerrainDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSTerrainCautionDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Terrain Caution'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainCautionDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSFailureDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Failure'
        self.phase_name = 'Airborne'
        self.node_class = TAWSFailureDuration
        self.values_mapping = {0: '-', 1: 'Failed'}

        self.basic_setup()


class TestTAWSObstacleWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Obstacle Warning'
        self.phase_name = 'Airborne'
        self.node_class = TAWSObstacleWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSPredictiveWindshearDuration(unittest.TestCase,
                                          CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Predictive Windshear'
        self.phase_name = 'Airborne'
        self.node_class = TAWSPredictiveWindshearDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSTerrainAheadDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Terrain Ahead'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainAheadDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSTerrainAheadPullUpDuration(unittest.TestCase,
                                         CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Terrain Ahead Pull Up'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainAheadPullUpDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSWindshearCautionBelow1500FtDuration(unittest.TestCase,
                                                  CreateKPVsWhereTest):
    def setUp(self):
        # TODO: remove after intervals have been implemented
        self.complex_where = True
        self.param_name = 'TAWS Windshear Caution'
        self.phase_name = 'Fast'
        self.node_class = TAWSWindshearCautionBelow1500FtDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.additional_params = [
            P(
                'Altitude AAL For Flight Phases',
                array=np.ma.array([
                    1501, 1502, 1501, 1499, 1498, 1499, 1499, 1499, 1499, 1501,
                    1502, 1501
                    ]),
            )
        ]

        self.basic_setup()


class TestTAWSWindshearSirenBelow1500FtDuration(unittest.TestCase,
                                                CreateKPVsWhereTest):
    def setUp(self):
        # TODO: remove after intervals have been implemented
        self.complex_where = True
        self.param_name = 'TAWS Windshear Siren'
        self.phase_name = 'Fast'
        self.node_class = TAWSWindshearSirenBelow1500FtDuration
        self.values_mapping = {0: '-', 1: 'Siren'}

        self.additional_params = [
            P(
                'Altitude AAL For Flight Phases',
                array=np.ma.array([
                    1501, 1502, 1501, 1499, 1498, 1499, 1499, 1499, 1499, 1501,
                    1502, 1501
                    ]),
            )
        ]

        self.basic_setup()


class TestTAWSWindshearWarningBelow1500FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSWindshearWarningBelow1500FtDuration
        self.operational_combinations = [('TAWS Windshear Warning', 'Altitude AAL For Flight Phases', 'Fast')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSUnspecifiedDuration(unittest.TestCase,
                                  CreateKPVsWhereTest):
    def setUp(self):
        self.param_name = 'TAWS Unspecified'
        self.phase_name = 'Airborne'
        self.node_class = TAWSUnspecifiedDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


##############################################################################
# Warnings: Traffic Collision Avoidance System (TCAS)


class TestTCASTAWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASTAWarningDuration
        self.operational_combinations = [('TCAS Traffic Advisory','TCAS Resolution Advisory'),]

    def test_derive(self):
        ta = buildsections('TCAS Traffic Advisory', [5, 8], [13, 17], [30, 39])
        ra = buildsection('TCAS Resolution Advisory', 9, 12)
        node = self.node_class()
        node.derive(ta, ra)
        self.assertEqual([KeyPointValue(30, 10, 'TCAS TA Warning Duration')],
                         node)

class TestTCASTAAcceleration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASTAAcceleration
        self.operational_combinations = [('Acceleration Vertical', 'TCAS Traffic Advisory', 'TCAS TA Warning Duration')]

    def test_derive(self):
        # For this simple test, acceleration sample rate = 1.0
        acc = P('Acceleration Vertical',
                array=np.ma.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.0, 1.0]))
        ta = buildsection('TCAS Traffic Advisory', 4, 8)
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=4, value=4)])
        node = self.node_class()
        node.derive(acc, ta, ta_warns)
        self.assertEqual(node[0].name, 'TCAS TA Acceleration', None)
        self.assertEqual(node[0].index, 7)
        self.assertAlmostEqual(node[0].value, 0.3)

    def test_negative(self):
        # For this simple test, acceleration sample rate = 1.0
        acc = P('Acceleration Vertical',
                array=np.ma.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0]))
        ta = buildsection('TCAS Traffic Advisory', 4, 8)
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=4, value=4)])
        node = self.node_class()
        node.derive(acc, ta, ta_warns)
        self.assertEqual(node[0].name, 'TCAS TA Acceleration', None)
        self.assertEqual(node[0].index, 6)
        self.assertAlmostEqual(node[0].value, -0.2)

    def test_nothing(self):
        # For this simple test, acceleration sample rate = 1.0
        acc = P('Acceleration Vertical',
                array=np.ma.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.09, 0.91, 1.0, 1.0]))
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=4, value=4)])
        ta = buildsection('TCAS Traffic Advisory', 4, 8)
        node = self.node_class()
        node.derive(acc, ta, ta_warns)
        self.assertAlmostEqual(node[0].value, 0.09)

    def test_not_after_ra(self):
        acc = P(name='Acceleration Vertical',
                array=np.ma.concatenate((np.ones(10), np.ones(5) * 1.2)))
        ta = buildsections('TCAS Traffic Advisory', [2, 5], [9,14])
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[])
        node = self.node_class()
        node.derive(acc, ta, ta_warns)
        self.assertEqual(node, [])


class TestTCASRAWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAWarningDuration
        self.operational_combinations = [('TCAS Resolution Advisory',)]

    def test_derive(self):
        ra = buildsection('TCAS Resolution Advisory', 5, 8)
        node = self.node_class()
        node.derive(ra)
        self.assertEqual([KeyPointValue(5.0, 3.0, 'TCAS RA Warning Duration')],
                         node)

class TestTCASRAWarningBelowFL100InClimbDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAWarningBelowFL100InClimbDuration
        self.operational_combinations = [('Altitude STD Smoothed',
                                          'TCAS Resolution Advisory')]

    def test_derive(self):
        testwave = (1.0 - np.cos(np.arange(0, 3.14 * 2, 0.05))) * 5100 + \
            (1.0 - np.cos(np.arange(0, 3.14 * 12, 0.3))) * 3000
        alt = P('Altitude STD Smoothed', array = testwave)
        # RA in first climb
        ra = buildsection('TCAS Resolution Advisory', 2, 24)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 22)
        self.assertEqual(node[0].name, 'TCAS RA Warning Below FL100 In Climb Duration')

        # RA in later climb
        ra = buildsection('TCAS Resolution Advisory', 43, 53)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 43)

        # RA in descent should not trigger this KPV
        ra = buildsection('TCAS Resolution Advisory', 84, 24)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 0)

        # RA above FL100 should not trigger this KPV
        ra = buildsection('TCAS Resolution Advisory', 60, 66)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 0)


class TestTCASRAWarningAboveFL100Duration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAWarningAboveFL100Duration
        self.operational_combinations = [('TCAS Resolution Advisory',
                                          'Altitude STD Smoothed')]

    def test_derive(self):
        testwave = (1.0 - np.cos(np.arange(0, 3.14 * 2, 0.05))) * 5100 + \
            (1.0 - np.cos(np.arange(0, 3.14 * 12, 0.3))) * 3000
        alt = P('Altitude STD Smoothed', array = testwave)
        ra = buildsection('TCAS Resolution Advisory', 32, 42)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 32)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[0].name, 'TCAS RA Warning Above FL100 Duration')

        # Not in climb
        ra = buildsection('TCAS Resolution Advisory', 3, 53)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 0)

        # Later period over FL100
        ra = buildsection('TCAS Resolution Advisory', 93, 103)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(node[0].index, 93)


class TestTCASRAWarningBelowFL100InDescentDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAWarningBelowFL100InDescentDuration
        self.operational_combinations = [('TCAS Resolution Advisory',
                                          'Altitude STD Smoothed')]

    def test_derive(self):
        testwave = (1.0 - np.cos(np.arange(0, 3.14 * 2, 0.05))) * 5100 + \
            (1.0 - np.cos(np.arange(0, 3.14 * 12, 0.3))) * 3000
        alt = P('Altitude STD Smoothed', array = testwave)
        # RA in first descent
        ra = buildsection('TCAS Resolution Advisory', 13, 33)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 20)
        self.assertEqual(node[0].name, 'TCAS RA Warning Below FL100 In Descent Duration')

        # RA in later descent
        ra = buildsection('TCAS Resolution Advisory', 120, 123)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 120)

        # RA in climb should not trigger this KPV
        ra = buildsection('TCAS Resolution Advisory', 23, 24)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 0)

        # RA above FL100 should not trigger this KPV
        ra = buildsection('TCAS Resolution Advisory', 60, 66)
        node = self.node_class()
        node.derive(alt, ra)
        self.assertEqual(len(node), 0)


class TestTCASRAReactionDelay(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAReactionDelay
        self.operational_combinations = [('Acceleration Vertical', 'TCAS Resolution Advisory'),
                                         ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Traffic Advisory')]

    def test_basic(self):
        # For this simple test, acceleration sample rate = 1.0
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(20), [1.1, 1.2, 1.3, 1.4, 1.5, 1.6], np.ones(4) * 1.7)))
        ra = buildsection('TCAS Resolution Advisory', 15, 30)
        node = self.node_class()
        node.derive(acc, ra, None)
        self.assertEqual(node[0].name, 'TCAS RA Reaction Delay')
        self.assertEqual(node[0].index, 15)
        self.assertAlmostEqual(node[0].value, 4.5, places=1)

    def test_ta_overlap(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(5), np.ones(15) * 1.2)))
        ra = buildsection('TCAS Resolution Advisory', 10, 20)
        ta = buildsection('TCAS Traffic Advisory', 3, 10)
        node = self.node_class()
        node.derive(acc, ra, ta)
        self.assertEqual(node[0].index, 10)
        self.assertAlmostEqual(node[0].value, -4.25, places=1)

    def test_not_second_ta_overlap(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(5), np.ones(15) * 1.2)))
        ra = buildsection('TCAS Resolution Advisory', 10, 16)
        ta = buildsections('TCAS Traffic Advisory', [3, 10], [16, 20])
        node = self.node_class()
        node.derive(acc, ra, ta)
        self.assertEqual(node[0].index, 10)
        self.assertAlmostEqual(node[0].value, -4.25, places=1)


class TestTCASRADirection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRADirection
        self.operational_combinations = [
            ('TCAS Resolution Advisory', 'TCAS Combined Control'),
            ('TCAS Resolution Advisory', 'TCAS Advisory Rate To Maintain'),
            ('TCAS Resolution Advisory', 'TCAS Altitude Rate To Maintain'),
            ('TCAS Resolution Advisory', 'TCAS Advisory Rate'),
            ('TCAS Resolution Advisory', 'TCAS Advisory Rate To Maintain',
             'TCAS Altitude Rate To Maintain', 'TCAS Advisory Rate'),
        ]
        self.values_mapping = {
            0: '-',
            1: 'Clear of Conflict',
            2: 'Drop Track',
            3: 'Altitude Lost',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'Preventive',
        }

    def test_cc_up(self):
        cc = M('TCAS Combined Control', array=np.ma.array([0,0,4,4,4,5,4,4,6,1,0,0]),
               values_mapping=self.values_mapping)
        ta = buildsection('TCAS Resolution Advisory', 2, 10)
        node = self.node_class()
        node.derive(ta, cc, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 1)

    def test_cc_down(self):
        cc = M('TCAS Combined Control', array=np.ma.array([0,0,5,5,5,5,4,4,6,1,0,0]),
               values_mapping=self.values_mapping)
        ta = buildsection('TCAS Resolution Advisory', 2, 10)
        node = self.node_class()
        node.derive(ta, cc, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, -1)

    def test_cc_prevent(self):
        cc = M('TCAS Combined Control', array=np.ma.array([0,0,6,6,6,6,6,4,6,1,0,0]),
               values_mapping=self.values_mapping)
        ta = buildsection('TCAS Resolution Advisory', 2, 10)
        node = self.node_class()
        node.derive(ta, cc, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 0)

    def test_rate_up(self):
        rate = P('TCAS Advisory Rate To Maintain', array=np.ma.array([0,0,1,0,0]))
        ta = buildsection('TCAS Resolution Advisory', 1, 3)
        for rates in ((rate, None, None),
                      (None, rate, None),
                      (None, None, rate),
                      (rate, rate, rate)):
            node = self.node_class()
            node.derive(ta, None, *rates)
            self.assertEqual(len(node), 1)
            self.assertEqual(node[0].value, 1)

    def test_rate_down(self):
        rate = P('TCAS Advisory Rate To Maintain', array=np.ma.array([1,1,0,1,1]))
        ta = buildsection('TCAS Resolution Advisory', 1, 3)
        node = self.node_class()
        node.derive(ta, None, rate, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, -1)

    def test_rate_unchanged(self):
        rate = P('TCAS Advisory Rate To Maintain', array=np.ma.ones(5))
        ta = buildsection('TCAS Resolution Advisory', 1, 3)
        node = self.node_class()
        node.derive(ta, None, rate, None, None)
        self.assertEqual(len(node), 0)


class TestTCASRAAcceleration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAAcceleration
        self.operational_combinations = [('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS RA Direction'),
                                         ('Acceleration Vertical', 'TCAS Traffic Advisory', 'TCAS Resolution Advisory', 'TCAS RA Direction')]

    def test_derive(self):
        # For this simple test, acceleration sample rate = 1.0
        acc = P('Acceleration Vertical',
                array=np.ma.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.45, 1.0, 1.0, 1.0]))
        ra = buildsection('TCAS Resolution Advisory', 2, 10)
        tcas_dir = KPV(name='TCAS Direction', items=[
            KeyPointValue(name='TCAS Direction', index=4, value=1)])
        node = self.node_class()
        node.derive(acc, None, ra, tcas_dir)
        self.assertEqual(node[0].name, 'TCAS RA Acceleration')
        self.assertEqual(node[0].index, 6)
        self.assertAlmostEqual(node[0].value, 0.45)


class TestTCASRAAPDisengaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAAPDisengaged
        self.operational_combinations = [('AP Disengaged Selection', 'TCAS Traffic Advisory', 'TCAS Resolution Advisory'),
                                         ('AP Disengaged Selection', 'TCAS Resolution Advisory')]

    def test_derive(self):
        kti_name = 'AP Disengaged Selection'
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(11, kti_name),
                                       KeyTimeInstance(27, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 22, 29)
        node = self.node_class()
        node.derive(ap_offs, tcas_ra, None)
        self.assertEqual(node,
                         [KeyPointValue(27.0, 1.0, 'TCAS RA AP Disengaged')])

    def test_no_disengagement(self):
        kti_name = 'AP Disengaged Selection'
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(11, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 22, 29)
        node = self.node_class()
        node.derive(ap_offs, tcas_ra, None)
        self.assertEqual(node,
                         [KeyPointValue(22.0, 0.0, 'TCAS RA AP Disengaged')])

    def test_long_ta(self):
        kti_name = 'AP Disengaged Selection'
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(11, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 22, 29)
        tcas_ta = buildsection('TCAS Traffic Advisory', 10, 20)
        node = self.node_class()
        node.derive(ap_offs, tcas_ra, tcas_ta)
        self.assertEqual(node,
                         [KeyPointValue(11.0, 1.0, 'TCAS RA AP Disengaged')])


class TestTCASRAToAPDisengagedDelay(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAToAPDisengagedDelay
        self.operational_combinations = [('AP Disengaged Selection', 'TCAS Resolution Advisory', 'Acceleration Vertical', 'TCAS Traffic Advisory'),
                                         ('AP Disengaged Selection', 'TCAS Resolution Advisory', 'Acceleration Vertical')]

    def test_derive(self):
        kti_name = 'AP Disengaged Selection'
        acc = P('Acceleration Vertical', array=np.ma.ones(10))
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(5, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 9)
        node = self.node_class()
        node.derive(acc, ap_offs, tcas_ra)
        self.assertEqual(node,
                         [KeyPointValue(3.0, 2.0, 'TCAS RA To AP Disengaged Delay')])

    def test_preemptive_disengagement_by_time(self):
        kti_name = 'AP Disengaged Selection'
        acc = P('Acceleration Vertical', array=np.ma.ones(100))
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(10, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 15, 70)
        node = self.node_class()
        node.derive(acc, ap_offs, tcas_ra)
        self.assertEqual(node,
                         [KeyPointValue(15.0, -5.0, 'TCAS RA To AP Disengaged Delay')])

    def test_preemptive_disengagement_by_ta(self):
        kti_name = 'AP Disengaged Selection'
        acc = P('Acceleration Vertical', array=np.ma.ones(10))
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(15, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 30, 70)
        tcas_ta = buildsection('TCAS Traffic Advisory', 10, 28)
        node = self.node_class()
        node.derive(acc, ap_offs, tcas_ra, tcas_ta)
        self.assertEqual(node,
                         [KeyPointValue(30.0, -15.0, 'TCAS RA To AP Disengaged Delay')])

    def test_no_disengagement(self):
        kti_name = 'AP Disengaged Selection'
        acc = P('Acceleration Vertical', array=np.ma.ones(10))
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(9, kti_name)])
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 7)
        node = self.node_class()
        node.derive(acc, ap_offs, tcas_ra)
        self.assertEqual(len(node), 0)


class TestTCASRAErroneousAcceleration (unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAErroneousAcceleration
        self.operational_combinations = [('Acceleration Vertical',
                                          'TCAS Resolution Advisory',
                                          'TCAS RA Direction'),
                                         ('Acceleration Vertical',
                                          'TCAS Traffic Advisory',
                                          'TCAS Resolution Advisory',
                                          'TCAS RA Direction'),
                                         ('Acceleration Vertical',
                                          'TCAS Resolution Advisory',
                                          'TCAS RA Acceleration',
                                          'TCAS RA Direction'),
                                         ('Acceleration Vertical',
                                          'TCAS Traffic Advisory',
                                          'TCAS Resolution Advisory',
                                          'TCAS RA Acceleration',
                                          'TCAS RA Direction')]

    def test_not_up(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(13), [0.95, 0.85, 0.6, 1.15, 1.2, 1.0, 1.0])))
        ra = buildsection('TCAS Resolution Advisory', 12, 20)
        tcas_dir = KPV(name='TCAS RA Direction', items=[
            KeyPointValue(name='TCAS RA Direction', index=13, value=1)])
        node = self.node_class()
        node.derive(acc, None, ra, None, tcas_dir)
        self.assertEqual(node[0].name, 'TCAS RA Erroneous Acceleration')
        self.assertEqual(node[0].index, 15)
        self.assertAlmostEqual(node[0].value, -0.4)

    def test_not_down(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(13), [1.05, 1.15, 1.6, 1.15, 0.2, 1.0, 1.0])))
        ra = buildsection('TCAS Resolution Advisory', 12, 20)
        tcas_dir = KPV(name='TCAS RA Direction', items=[
            KeyPointValue(name='TCAS RA Direction', index=13, value=-1)])
        node = self.node_class()
        node.derive(acc, None, ra, None, tcas_dir)
        self.assertEqual(node[0].name, 'TCAS RA Erroneous Acceleration')
        self.assertEqual(node[0].index, 15)
        self.assertAlmostEqual(node[0].value, 0.6)

    def test_neither_up_nor_down(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(13), [0.95, 0.85, 0.6, 1.15, 1.2, 1.0, 1.0])))
        ra = buildsection('TCAS Resolution Advisory', 12, 20)
        tcas_dir = KPV(name='TCAS RA Direction', items=[
            KeyPointValue(name='TCAS RA Direction', index=13, value=0)])
        node = self.node_class()
        node.derive(acc, None, ra, None, tcas_dir)
        self.assertEqual(node, [])

    def test_ta_overlap(self):
        acc = P('Acceleration Vertical',
                array=np.ma.concatenate((np.ones(15), [1.2, 1.2, 1.35], np.ones(12))))
        ra = buildsection('TCAS Resolution Advisory', 20, 30)
        ta = buildsection('TCAS Traffic Advisory', 3, 18)
        tcas_dir = KPV(name='TCAS RA Direction', items=[
            KeyPointValue(name='TCAS RA Direction', index=20, value=-1)])
        node = self.node_class()
        node.derive(acc, ta, ra, None, tcas_dir)
        self.assertEqual(node[0].index, 17)
        self.assertAlmostEqual(node[0].value, 0.35)


class TestTCASRASubsequentReactionDelay (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASRASubsequentReactionDelay
        self.operational_combinations = [
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Advance Rate To Maintain'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Advisory Rate To Maintain'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Altitude Rate To Maintain'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Advisory Rate'),
        ]
        self.values_mapping = {
            0: 'None',
            1: 'Clear of Conflict',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'Preventive',
        }

    def test_rate_advisory_data_up(self):
        acc = P('Acceleration Vertical', array=np.ma.concatenate((np.ones(15), np.ones(5) * 1.2)))
        ta = buildsection('TCAS Resolution Advisory', 5, 20)
        cc = M('TCAS Combined Control', array=np.ma.ones(20) * 4, values_mapping=self.values_mapping)
        rate_1 = None #TCAS Advance Rate To Maintain
        rate_2 = None #TCAS Advisory Rate To Maintain
        rate_3 = P('TCAS Altitude Rate To Maintain', array=np.ma.concatenate((np.zeros(10), np.ones(10) * 1000)))
        rate_4 = None #TCAS Advisory Rate
        node = self.node_class()
        node.derive(acc, ta, cc, rate_1, rate_2, rate_3, rate_4)
        self.assertEqual(node[0].name, 'TCAS RA Subsequent Reaction Delay')
        self.assertEqual(node[0].index, 9)
        self.assertAlmostEqual(node[0].value, 6.75)

    def test_rate_advisory_data_down(self):
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*15+[0.8]*5))
        ta = buildsection('TCAS Resolution Advisory', 5, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([4]*20),
               values_mapping=self.values_mapping)
        rate_1 = None #TCAS Advance Rate To Maintain
        rate_2 = P('TCAS Advisory Rate To Maintain', array=np.ma.array([0]*10 + [-1000]*10))
        rate_3 = None #TCAS Altitude Rate To Maintain
        rate_4 = None #TCAS Advisory Rate
        node = self.node_class()
        node.derive(acc, ta, cc, rate_1, rate_2, rate_3, rate_4)
        self.assertEqual(node[0].name, 'TCAS RA Subsequent Reaction Delay')
        self.assertEqual(node[0].index, 9)
        self.assertAlmostEqual(node[0].value, 6.75)

    def test_combined_control_data_up(self):
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*15+[1.2]*5))
        ta = buildsection('TCAS Resolution Advisory', 2, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([0]*2 + [6]*8 + [4]*10),
               values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(acc, ta, cc, None, None, None, None)
        self.assertEqual(node[0].name, 'TCAS RA Subsequent Reaction Delay')
        self.assertEqual(node[0].index, 9)
        self.assertAlmostEqual(node[0].value, 6.75)

    def test_combined_control_data_down(self):
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*15+[0.8]*5))
        ta = buildsection('TCAS Resolution Advisory', 2, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([0]*2 + [6]*8 + [5]*10),
               values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(acc, ta, cc, None, None, None, None)
        self.assertEqual(node[0].name, 'TCAS RA Subsequent Reaction Delay')
        self.assertEqual(node[0].index, 9)
        self.assertAlmostEqual(node[0].value, 6.75)

    def test_combined_control_data_clear(self):
        acc = P('Acceleration Vertical', array=np.ma.concatenate((np.ones(15), np.ones(5) * 0.8)))
        ta = buildsection('TCAS Resolution Advisory', 2, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([0]*2 + [6]*8 + [1]*10),
               values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(acc, ta, cc, None, None, None, None)
        self.assertEqual(node, [])

    def test_combined_control_no_change(self):
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*15+[0.8]*5))
        ta = buildsection('TCAS Resolution Advisory', 2, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([0]*2 + [6]*18),
               values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(acc, ta, cc, None, None, None, None)
        self.assertEqual(node, [])

    def test_no_reaction(self):
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*20))
        ta = buildsection('TCAS Resolution Advisory', 2, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([0]*2 + [6]*8 + [4]*10),
               values_mapping=self.values_mapping)
        rate_1 = None #TCAS Altitude Rate Advisory
        rate_2 = None #TCAS Advisory Rate To Maintain
        rate_3 = None #TCAS Altitude Rate To Maintain
        rate_4 = None #TCAS Advisory Rate
        node = self.node_class()
        node.derive(acc, ta, cc, rate_1, rate_2, rate_3, rate_4)
        self.assertEqual(node, [])


class TestTCASRASubsequentAcceleration (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASRASubsequentAcceleration
        self.operational_combinations = [
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Advisory Rate To Maintain'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Altitude Rate To Maintain'),
            ('Acceleration Vertical', 'TCAS Resolution Advisory', 'TCAS Combined Control', 'TCAS Advisory Rate'),
        ]
        self.values_mapping = {
            0: 'None',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'Preventive'}

    def test_triggered_by_rate_advisory_data(self):
        # Also demonstrate that any of the four advisory rate parameters can operate
        acc = P('Acceleration Vertical', array=np.ma.array([1.0]*15+[0.8, 0.7, 0.6, 0.7, 0.8]))
        ta = buildsection('TCAS Resolution Advisory', 5, 20)
        cc = M('TCAS Combined Control', array=np.ma.array([4]*20),
               values_mapping=self.values_mapping)
        rate_2 = None #TCAS Advisory Rate To Maintain
        rate_3 = P('TCAS Altitude Rate To Maintain', array=np.ma.array([0]*10 + [1000]*10))
        rate_4 = None #TCAS Advisory Rate
        node = self.node_class()
        node.derive(acc, ta, cc, rate_2, rate_3, rate_4)
        self.assertEqual(node[0].name, 'TCAS RA Subsequent Acceleration')
        self.assertEqual(node[0].index, 17)
        self.assertAlmostEqual(node[0].value, -0.4)


class TestTCASRAChangeOfVerticalSpeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAChangeOfVerticalSpeed
        self.operational_combinations = [('Vertical Speed', 'TCAS Resolution Advisory')]

    def test_derive_downward(self):
        vs=P('Vertical Speed', array=np.ma.array([0]*5+[200,-300]))
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 8)
        node = self.node_class()
        node.derive(vs, tcas_ra)
        self.assertEqual(node.name, 'TCAS RA Change Of Vertical Speed')
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, -300.0)

    def test_derive_upward(self):
        vs=P('Vertical Speed', array=np.ma.array([-100]*5+[-200,300]))
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 8)
        node = self.node_class()
        node.derive(vs, tcas_ra)
        self.assertEqual(node[0].value, 400.0)


class TestTCASRAAltitudeSTD (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASRAAltitudeSTD
        self.operational_combinations = [('Altitude STD', 'TCAS Resolution Advisory')]

    def test_derive(self):
        alt=P('Altitude STD', array=np.ma.array(range(15000, 15500, 100)))
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 8)
        node = self.node_class()
        node.derive(alt, tcas_ra)
        self.assertEqual(node.name, 'TCAS RA Altitude STD')
        self.assertEqual(node[0].value, 15300)


class TestTCASRAAltitudeAAL (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASRAAltitudeAAL
        self.operational_combinations = [('Altitude AAL', 'TCAS Resolution Advisory')]

    def test_derive(self):
        alt=P('Altitude AAL', array=np.ma.array(range(2000,3000,100)))
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 8)
        node = self.node_class()
        node.derive(alt, tcas_ra)
        self.assertEqual(node.name, 'TCAS RA Altitude AAL')
        self.assertEqual(node[0].value, 2300)


class TestTCASRAHeading (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASRAHeading
        self.operational_combinations = [('Heading', 'TCAS Resolution Advisory')]

    def test_derive(self):
        hdg=P('Altitude AAL', array=np.ma.array(range(0, 360, 10)))
        tcas_ra = buildsection('TCAS Resolution Advisory', 3, 8)
        node = self.node_class()
        node.derive(hdg, tcas_ra)
        self.assertEqual(node.name, 'TCAS RA Heading')
        self.assertEqual(node[0].value, 30)


class TestTCASTAAltitudeSTD (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASTAAltitudeSTD
        self.operational_combinations = [('Altitude STD', 'TCAS TA Warning Duration')]

    def test_derive(self):
        alt=P('Altitude STD', array=np.ma.array(range(15000, 15500, 100)))
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=3, value=0.4)])
        node = self.node_class()
        node.derive(alt, ta_warns)
        self.assertEqual(node.name, 'TCAS TA Altitude STD')
        self.assertEqual(node[0].value, 15300)

    def test_not_for_ra_only(self):
        alt=P('Altitude STD', array=np.ma.array(range(15000, 15500, 100)))
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[])
        node = self.node_class()
        node.derive(alt, ta_warns)
        self.assertEqual(node, [])


class TestTCASTAAltitudeAAL (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASTAAltitudeAAL
        self.operational_combinations = [('Altitude AAL', 'TCAS TA Warning Duration')]

    def test_derive(self):
        alt=P('Altitude AAL', array=np.ma.array(range(2000,3000,100)))
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=3, value=0.4)])
        node = self.node_class()
        node.derive(alt, ta_warns)
        self.assertEqual(node.name, 'TCAS TA Altitude AAL')
        self.assertEqual(node[0].value, 2300)


class TestTCASTAHeading (unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASTAHeading
        self.operational_combinations = [('Heading', 'TCAS TA Warning Duration')]

    def test_derive(self):
        hdg=P('Altitude AAL', array=np.ma.array(range(0, 360, 10)))
        ta_warns = KPV(name='TCAS TA Warning Duration', items=[
            KeyPointValue(name='TCAS TA Warning Duration', index=3, value=0.4)])
        node = self.node_class()
        node.derive(hdg, ta_warns)
        self.assertEqual(node.name, 'TCAS TA Heading')
        self.assertEqual(node[0].value, 30)


class TestTCASFailureRatio(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TCASFailureRatio
        self.values_mapping = {0: '-', 1: 'Failed'}
        self.operational_combinations = [('TCAS Operational', 'Airborne')]

    def test_normal(self):
        tcas = buildsection('TCAS Operational', 6, 14)
        airs = buildsection('Airborne', 5, 14)
        node = self.node_class()
        node.derive(tcas, airs)
        self.assertEqual(node[0].name, 'TCAS Failure Ratio')
        self.assertEqual(node[0].index, 0)
        self.assertAlmostEqual(node[0].value, 10)

    def test_divide_zero(self):
        tcas = buildsection('TCAS Operational', 1, 0)
        airs = buildsection('Airborne', 1, 0)
        node = self.node_class()
        node.derive(tcas, airs)
        self.assertEqual(node[0].name, 'TCAS Failure Ratio')
        self.assertEqual(node[0].index, 0)
        self.assertAlmostEqual(node[0].value, 0)


##############################################################################
# Warnings: Takeoff Configuration


class TestTakeoffConfigurationWarningDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = TakeoffConfigurationWarningDuration

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            manufacturer=A('Manufacturer', 'Airbus')), [])
        self.assertEqual(self.node_class.get_operational_combinations(
            manufacturer=A('Manufacturer', 'Boeing')),
                         [('Takeoff Configuration Warning', 'Movement Start', 'Liftoff')])

    def test_derive(self):
        node = self.node_class()
        takeoff_warn = M('Takeoff Configuration Warning',
                         np.ma.array([1,0,0,1,1,0,0,0,0,1,1]),
                         values_mapping={1: 'Warning', 0: '-'})
        movement_starts = KTI('Liftoff', items=[KeyTimeInstance(2, 'Movement Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        node.derive(takeoff_warn, movement_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)


class TestTakeoffConfigurationFlapWarningDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffConfigurationFlapWarningDuration
        self.operational_combinations = [('Takeoff Configuration Flap Warning', 'Movement Start', 'Liftoff')]

    def test_derive(self):
        node = self.node_class()
        takeoff_warn = M('Takeoff Configuration Flap Warning',
                         np.ma.array([1,0,0,1,1,0,0,0,0,1,1]),
                         values_mapping={1: 'Warning', 0: '-'})
        movement_starts = KTI('Liftoff', items=[KeyTimeInstance(2, 'Movement Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        node.derive(takeoff_warn, movement_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)


class TestTakeoffConfigurationParkingBrakeWarningDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffConfigurationParkingBrakeWarningDuration
        self.operational_combinations = [('Takeoff Configuration Parking Brake Warning', 'Movement Start', 'Liftoff')]

    def test_derive(self):
        node = self.node_class()
        takeoff_warn = M('Takeoff Configuration Parking Brake Warning',
                         np.ma.array([1,0,0,1,1,0,0,0,0,1,1]),
                         values_mapping={1: 'Warning', 0: '-'})
        movement_starts = KTI('Liftoff', items=[KeyTimeInstance(2, 'Movement Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        node.derive(takeoff_warn, movement_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)


class TestTakeoffConfigurationSpoilerWarningDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffConfigurationSpoilerWarningDuration
        self.operational_combinations = [('Takeoff Configuration Spoiler Warning', 'Movement Start', 'Liftoff')]

    def test_derive(self):
        node = self.node_class()
        takeoff_warn = M('Takeoff Configuration Spoiler Warning',
                         np.ma.array([1,0,0,1,1,0,0,0,0,1,1]),
                         values_mapping={1: 'Warning', 0: '-'})
        movement_starts = KTI('Liftoff', items=[KeyTimeInstance(2, 'Movement Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        node.derive(takeoff_warn, movement_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)


class TestTakeoffConfigurationStabilizerWarningDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffConfigurationStabilizerWarningDuration
        self.operational_combinations = [('Takeoff Configuration Stabilizer Warning', 'Movement Start', 'Liftoff')]

    def test_derive(self):
        node = self.node_class()
        takeoff_warn = M('Takeoff Configuration Stabilizer Warning',
                         np.ma.array([1,0,0,1,1,0,0,0,0,1,1]),
                         values_mapping={1: 'Warning', 0: '-'})
        movement_starts = KTI('Liftoff', items=[KeyTimeInstance(2, 'Movement Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'Liftoff')])
        node.derive(takeoff_warn, movement_starts, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 2)


##############################################################################
# Warnings: Smoke


class TestSmokeWarningDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = SmokeWarningDuration

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [
            ('Smoke Warning',)])

    def test_derive(self):
        smoke = MultistateDerivedParameterNode(
            name='Smoke Warning',
            array=np.ma.array([0] * 5 + [1] * 10 + [0] * 15),
            values_mapping={0: '-', 1: 'Smoke'}
        )
        node = self.node_class()
        node.derive(smoke)

        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[0].index, 5)


##############################################################################
# Warnings: Smoke and Fire

class TestCargoSmokeOrFireWarningDuration(unittest.TestCase, CreateKPVsWhereTest):

    def setUp(self):
        self.param_name = 'Cargo (*) Smoke Or Fire'
        self.phase_name = 'Airborne'
        self.node_class = CargoSmokeOrFireWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


##############################################################################
# Throttle


class TestThrottleCyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrottleCyclesDuringFinalApproach
        self.operational_combinations = [('Throttle Levers', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrottleLeverAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = ThrottleLeverAtLiftoff
        self.operational_combinations = [('Throttle Levers', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrottleLeverVariationAbove80KtsToTakeoff(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ThrottleLeverVariationAbove80KtsToTakeoff
        self.operational_combinations = [(
            'Throttle Levers',
            'Airspeed',
            'Takeoff Roll Or Rejected Takeoff',
        )]

    def test_derive(self):
        '''
        Three takeoffs, 1st is 4 degree increase, 2nd has no change and
        3rd is -4 degree decrease.
        '''
        t_array = [0, 0, 24, 24, 66] + [66]*25 + [70]*25 + [47]*5 + \
            [0, 0, 24, 24, 66] + [66]*50 + [47]*5 + \
            [0, 0, 24, 24, 66] + [66]*25 + [62]*25 + [47]*5
        levers = P('Throttle Lever', np.ma.array(t_array))
        a_array = ([0.]*5 + np.linspace(0, 170, 50).tolist() + [170.]*5)*3
        airspeed = P('Airspeed', np.ma.array(a_array))
        takeoff = buildsections('Takeoff Roll Or Rejected Takeoff',
                                [2, 54], [62, 114], [122, 174])

        node = self.node_class()
        node.derive(levers, airspeed, takeoff)

        self.assertEqual(len(node), 3)
        self.assertAlmostEqual(node[0].index, 30, 0)
        self.assertAlmostEqual(node[0].value, 4, 0)
        self.assertAlmostEqual(node[1].index, 88, 0)
        self.assertAlmostEqual(node[1].value, 0, 0)
        self.assertAlmostEqual(node[2].index, 150, 0)
        self.assertAlmostEqual(node[2].value, -4, 0)


##############################################################################
# Thrust Asymmetry


class TestThrustAsymmetryDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringTakeoffMax
        self.operational_combinations = [('Thrust Asymmetry',
                                          'Takeoff Roll Or Rejected Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringFlightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringFlightMax
        self.operational_combinations = [('Thrust Asymmetry', 'Airborne')]

    def test_derive(self):
        ta = P(name='Thrust Asymmetry', array=np.ma.array([0]*400 + [29]*350 + [0]*5 + [30]*301 + [0]*50 + [31]*299 + [0]*95), dtype=float)
        air = buildsection('Airborne', 100, 1450)
        node = self.node_class()
        node.derive(ta, air)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 31)


class TestThrustAsymmetryDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringGoAroundMax
        self.operational_combinations = [('Thrust Asymmetry', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringApproachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringApproachMax
        self.operational_combinations = [('Thrust Asymmetry', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryWithThrustReversersDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryWithThrustReversersDeployedMax
        self.operational_combinations = [('Thrust Asymmetry', 'Thrust Reversers', 'Mobile')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringApproachDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringApproachDuration
        self.operational_combinations = [('Thrust Asymmetry', 'Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryWithThrustReversersDeployedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryWithThrustReversersDeployedDuration
        self.operational_combinations = [('Thrust Asymmetry', 'Thrust Reversers', 'Mobile')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestTouchdownToElevatorDownDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TouchdownToElevatorDownDuration
        self.operational_combinations = [('Airspeed', 'Elevator', 'Touchdown', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownTo60KtsDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TouchdownTo60KtsDuration
        self.operational_combinations = [
            ('Airspeed', 'Touchdown'),
            ('Airspeed', 'Groundspeed', 'Touchdown'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownToPitch2DegreesAbovePitchAt60KtsDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = TouchdownToPitch2DegreesAbovePitchAt60KtsDuration

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(len(opts), 1)
        self.assertIn('Pitch', opts[0])
        self.assertIn('Airspeed', opts[0])
        self.assertIn('Touchdown', opts[0])

    def test_derive(self):
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(8,'Touchdown'), ])
        airspeed = P('Airspeed', np.ma.array(
            [150, 146, 142, 145, 143, 139, 142, 137, 133, 140,
             133, 128, 131, 128, 124, 120, 124, 123, 120, 118,
             113, 110, 112, 107, 108, 106, 101, 97, 95, 94,
             90, 86, 86, 84, 80, 79, 78, 73, 73, 68,
             64, 60, 58, 51, 47, 43, 46, 44, 40, 43,],
            mask=[False]*41 + [True]*9
        ))
        pitch = P('Pitch', np.ma.array(
            [2.8, 3.3, 4.7, 4.7, 4.4, 5.1, 5.1, 4.9, 5.8, 5.4,
             6.0, 6.0, 5.8, 6.9, 7.0, 7.6, 7.4, 7.2, 7.7, 7.9,
             8.1, 8.4, 8.8, 9.3, 9.0, 8.6, 7.9, 7.7, 7.2, 5.8,
             4.7, 3.7, 3.0, 2.1, 1.2, -0.2, -0.5, -0.7, -0.9, -0.9,
             -0.7, -0.9, -1.1, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,]
        ))

        node = self.node_class()
        node.derive(pitch, airspeed, tdwns)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 34.1, places=1)
        self.assertAlmostEqual(node[0].value, 26.1, places=1)

    def test_derive_pitch_not_2_deg_higher(self):
        # From pti-90dbc326da5a after touchdown the pitch never went above the
        # 2 degrees (0 deg at 60Kts), This test should not produce a KPV.
        airspeed = load(os.path.join(test_data_path,
                                     'pti-90dbc326da5a-airspeed.nod'))
        pitch = load(os.path.join(test_data_path,
                                  'pti-90dbc326da5a-pitch.nod'))
        tdwns = KTI('Touchdown',
                    items=[KeyTimeInstance(5760.140625,'Touchdown'), ])

        node = self.node_class()
        node.derive(pitch, airspeed, tdwns)

        self.assertAlmostEqual(len(node), 0)

##############################################################################
# Turbulence


class TestTurbulenceDuringApproachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TurbulenceDuringApproachMax
        self.operational_combinations = [('Turbulence', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTurbulenceDuringCruiseMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TurbulenceDuringCruiseMax
        self.operational_combinations = [('Turbulence', 'Cruise')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTurbulenceDuringFlightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TurbulenceDuringFlightMax
        self.operational_combinations = [('Turbulence', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Wind


class TestWindSpeedAtAltitudeDuringDescent(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindSpeedAtAltitudeDuringDescent
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Wind Speed')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirectionAtAltitudeDuringDescent(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindDirectionAtAltitudeDuringDescent
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Wind Direction Continuous')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindAcrossLandingRunwayAt50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindAcrossLandingRunwayAt50Ft
        self.operational_combinations = [('Wind Across Landing Runway', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailwindAtAltitudeDuringDescent(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TailwindAtAltitudeDuringDescent
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Tailwind')]

    def test_derive(self):
        alt_aal = P(
            'Altitude AAL For Flight Phases',
            array=np.ma.array(data=np.linspace(2500, 0, num=51))
        )
        tailwind = P(
            'Tailwind',
            array=alt_aal.array / 50.0
        )
        node = self.node_class()
        node.derive(alt_aal, tailwind)
        for alt in (2000, 1500, 1000, 500, 100, 50):
            kpvs = node.get(name='Tailwind At {} Ft During Descent'.format(alt))
            self.assertEqual(len(kpvs), 1)
            kpv = kpvs[0]
            self.assertAlmostEqual(kpv.value, alt / 50.0)


##############################################################################
# Weight


class TestGrossWeightAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight Smoothed', 'Liftoff')]
        self.gw = P(name='Gross Weight Smoothed', array=np.ma.array((1, 2, 3)))
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=1),
        ])

    def test_derive__basic(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.gw, self.liftoffs)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=1, value=2),
        ]))

    def test_derive__masked(self):
        self.gw.array.mask = True
        node = self.node_class()
        node.derive(self.gw, self.liftoffs)
        self.assertEqual(len(node), 0)


class TestGrossWeightAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GrossWeightAtTouchdown
        self.operational_combinations = [('Gross Weight Smoothed', 'Touchdown', 'Touch And Go')]
        self.gw = P(name='Gross Weight Smoothed', array=np.ma.array((1, 2, 3)))
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=2),
        ])
        self.touch_and_go = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touch And Go', index=1),
        ])

    def test_derive__basic(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.gw, self.touchdowns)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=2, value=3),
        ]))

    def test_derive__masked(self):
        self.gw.array.mask = True
        node = self.node_class()
        node.derive(self.gw, self.touchdowns)
        self.assertEqual(len(node), 0)

    def test_derive__touch_and_go(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.gw, self.touch_and_go)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(name=name, index=1, value=2),
        ]))


class TestGrossWeightConditionalAtTouchdown(unittest.TestCase):

    def setUp(self):
        self.name = 'Gross Weight Conditional At Touchdown'
        self.node_class = GrossWeightConditionalAtTouchdown
        self.manufacturer = A('Manufacturer', 'Airbus')
        self.weight = KPV(name='Gross Weight At Touchdown', items=[
            KeyPointValue(name='Gross Weight At Touchdown', index=6107, value=109301),
        ])
        self.accel = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(name='Acceleration Normal At Touchdown', index=6107, value=1.8),
        ])
        self.rod = KPV(name='Rate Of Descent At Touchdown', items=[
            KeyPointValue(name='Rate Of Descent At Touchdown', index=6107, value=-400),
        ])
        self.expected = KPV(name=self.name, items=[
            KeyPointValue(name=self.name, index=6107, value=109301),
        ])


    def test_can_operate(self):
        available = ('Gross Weight At Touchdown',
                     'Acceleration Normal At Touchdown',
                     'Rate Of Descent At Touchdown')
        self.assertTrue(self.node_class().can_operate(available, manufacturer=self.manufacturer))
        self.assertFalse(self.node_class().can_operate(('Gross Weight At Touchdown',), manufacturer=self.manufacturer))
        self.manufacturer.value = 'Boeing'
        self.assertFalse(self.node_class().can_operate(available, manufacturer=self.manufacturer))


    def test_derive(self):
        node = self.node_class()
        node.derive(self.weight, self.accel, self.rod)

        self.assertEqual(len(node), 1)
        self.assertEqual(node, self.expected)


    def test_derive__hi_g(self):
        self.rod[0].value = 340

        node = self.node_class()
        node.derive(self.weight, self.accel, self.rod)


        self.assertEqual(len(node), 1)
        self.assertEqual(node, self.expected)


    def test_derive__hi_rod(self):
        self.accel[0].value = 1.6

        node = self.node_class()
        node.derive(self.weight, self.accel, self.rod)

        self.assertEqual(len(node), 1)
        self.assertEqual(node, self.expected)


    def test_derive__gentle_touchdown(self):
        self.accel[0].value = 1.6
        self.rod[0].value = -340

        node = self.node_class()
        node.derive(self.weight, self.accel, self.rod)

        self.assertEqual(len(node), 0)

    def test_derive_tdwn_idx_rounding(self):
        weight = KPV(name='Gross Weight At Touchdown', items=[
            KeyPointValue(name='Gross Weight At Touchdown',
                          index=7999.125, value=9256.642893),
        ])
        accel = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(name='Acceleration Normal At Touchdown',
                          index=8000.1875, value=1.76901324178),
        ])
        rod = KPV(name='Rate Of Descent At Touchdown', items=[
            KeyPointValue(name='Rate Of Descent At Touchdown',
                          index=7999.125, value=-636.019657353),
        ])

        node = self.node_class()
        node.derive(weight, accel, rod)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 7999.125)
        self.assertAlmostEquals(node[0].value, 9256.64, 2)

    def test_derive_tdwn_idx_rounding_2(self):
        weight = KPV(name='Gross Weight At Touchdown', items=[
            KeyPointValue(name='Gross Weight At Touchdown',
                          index=8454.125, value=61687.7198037),
        ])
        accel = KPV(name='Acceleration Normal At Touchdown', items=[
            KeyPointValue(name='Acceleration Normal At Touchdown',
                          index=8455.3125, value=1.71677918365),
        ])
        rod = KPV(name='Rate Of Descent At Touchdown', items=[
            KeyPointValue(name='Rate Of Descent At Touchdown',
                          index=8454.125, value=-20.5970559871),
        ])

        node = self.node_class()
        node.derive(weight, accel, rod)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8454.125)
        self.assertAlmostEquals(node[0].value, 61687.72, 2)


class TestGrossWeightDelta60SecondsInFlightMax(unittest.TestCase):

    def test_can_operate(self):
        opts = GrossWeightDelta60SecondsInFlightMax.get_operational_combinations()
        self.assertEqual(opts, [('Gross Weight', 'Airborne')])

    def test_gross_weight_delta_superframe(self):
        # simulate a superframe recorded parameter
        weight = P('Gross Weight', [-10,2,3,4,6,7,8],
                   frequency=1/64.0)
        airborne = buildsection('Airborne', 100, None)
        gwd = GrossWeightDelta60SecondsInFlightMax()
        gwd.get_derived([weight, airborne])
        self.assertEqual(len(gwd), 1)
        self.assertEqual(gwd[0].index, 239)
        self.assertEqual(gwd[0].value, 1.6875)

    def test_gross_weight_delta_1hz(self):
        # simulate a superframe recorded parameter
        weight = P('Gross Weight', np.ma.repeat([-10,2,3,4,6,7,8,9,10,11,12,13,
                                                 14,15,16,17,18,19,20,21,22,23,
                                                 24,25,26,27,28,29,30], 11),
                   frequency=1)
        airborne = buildsection('Airborne', 100, None)
        gwd = GrossWeightDelta60SecondsInFlightMax()
        gwd.get_derived([weight, airborne])
        self.assertEqual(len(gwd), 1)
        self.assertEqual(gwd[0].index, 176)
        self.assertEqual(gwd[0].value, 6)


##############################################################################
# Dual Input


class TestDualInputWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DualInputWarningDuration
        self.operational_combinations = [
            ('Dual Input Warning', 'Takeoff Roll', 'Landing Roll'),
        ]

    def test_derive(self):
        mapping = {0: '-', 1: 'Dual'}
        dual = M('Dual Input Warning', np.ma.zeros(50), values_mapping=mapping)
        dual.array[3:10] = 'Dual'

        takeoff_roll = buildsection('Takeoff Roll', 0, 5)
        landing_roll = buildsection('Landing Roll', 44, 50)

        node = self.node_class()
        node.derive(dual, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=3, value=7.0),
        ])

        self.assertEqual(node, expected)

    def test_derive_from_hdf(self):
        path = os.path.join(test_data_path, 'dual_input.hdf5')
        [dual], phase = self.get_params_from_hdf(path, ['Dual Input Warning'])
        dual.name = 'Dual Input Warning'  # Hack...

        takeoff_roll = buildsection('Takeoff Roll', 0, 100)
        landing_roll = buildsection('Landing Roll', 320, 420)

        node = self.node_class()
        node.derive(dual, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=91, value=31.0),
            KeyPointValue(name=name, index=213, value=59.0),
        ])

        self.assertEqual(node, expected)


class TestDualInputAbove200FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DualInputAbove200FtDuration
        self.operational_combinations = [
            ('Dual Input', 'Altitude AAL'),
        ]

    def test_derive(self):
        mapping = {0: '-', 1: 'Dual'}
        dual = M('Dual Input', np.ma.zeros(50), values_mapping=mapping)
        dual.array[3:10] = 'Dual'

        alt_aal = P('Altitude AAL', np.ma.arange(50) * 35)

        node = self.node_class()
        node.derive(dual, alt_aal)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=6, value=4.0),
        ])

        self.assertEqual(node, expected)

    def test_derive_from_hdf(self):
        path = os.path.join(test_data_path, 'dual_input.hdf5')
        [alt_aal], phase = self.get_params_from_hdf(path,
                                                    ['Altitude AAL'])

        mapping = {0: '-', 1: 'Dual'}
        dual_array = MappedArray(
            np.ma.zeros(alt_aal.array.size),
        values_mapping=mapping)
        dual_array[91:122] = 'Dual'
        dual_array[213:272] = 'Dual'
        dual = M('Dual Input', dual_array, values_mapping=mapping)

        node = self.node_class()
        node.derive(dual, alt_aal)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=256, value=16.0),
        ])

        self.assertEqual(node, expected)


class TestDualInputBelow200FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DualInputBelow200FtDuration
        self.operational_combinations = [
            ('Dual Input', 'Altitude AAL', 'Takeoff Roll', 'Landing Roll'),
        ]

    def test_derive(self):
        mapping = {0: '-', 1: 'Dual'}
        dual = M('Dual Input', np.ma.zeros(50), values_mapping=mapping)
        dual.array[3:10] = 'Dual'

        alt_aal = P('Altitude AAL', np.ma.arange(50)[::-1] * 5)

        takeoff_roll = buildsection('Takeoff Roll', 0, 5)
        landing_roll = buildsection('Landing Roll', 44, 50)

        node = self.node_class()
        node.derive(dual, alt_aal, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=9, value=1.0),
        ])

        self.assertEqual(node, expected)

    def test_derive_from_hdf(self):
        path = os.path.join(test_data_path, 'dual_input.hdf5')
        [alt_aal], phase = self.get_params_from_hdf(path,
                                                    ['Altitude AAL'])

        takeoff_roll = buildsection('Takeoff Roll', 0, 100)
        landing_roll = buildsection('Landing Roll', 320, 420)

        # Dual Input parameter calculated from Dual Input hdf file
        mapping = {0: '-', 1: 'Dual'}
        dual_array = MappedArray(
            np.ma.zeros(alt_aal.array.size),
            values_mapping=mapping)
        dual_array[91:122] = 'Dual'
        dual_array[213:272] = 'Dual'
        dual = M('Dual Input', dual_array, values_mapping=mapping)

        node = self.node_class()
        node.derive(dual, alt_aal, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=91, value=31.0),
            KeyPointValue(name=name, index=213, value=43.0),
        ])

        self.assertEqual(node, expected)


class TestDualInputByCaptDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DualInputByCaptDuration
        self.operational_combinations = [
            ('Dual Input', 'Pilot Flying', 'Takeoff Roll', 'Landing Roll'),
        ]

    def test_derive(self):
        mapping = {0: '-', 1: 'Dual'}
        dual = M('Dual Input', np.ma.zeros(50), values_mapping=mapping)
        dual.array[3:10] = 'Dual'

        mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}
        pilot = M('Pilot Flying', np.ma.zeros(50), values_mapping=mapping)
        pilot.array[0:20] = 'First Officer'

        takeoff_roll = buildsection('Takeoff Roll', 0, 5)
        landing_roll = buildsection('Landing Roll', 44, 50)

        node = self.node_class()
        node.derive(dual, pilot, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=3, value=7.0),
        ])

        self.assertEqual(node, expected)

    @unittest.skip('No data available in the relevant HDF for this test case.')
    def test_derive_from_hdf(self):
        pass


class TestDualInputByFODuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DualInputByFODuration
        self.operational_combinations = [
            ('Dual Input', 'Pilot Flying', 'Takeoff Roll', 'Landing Roll'),
        ]

    def test_derive(self):
        mapping = {0: '-', 1: 'Dual'}
        dual = M('Dual Input', np.ma.zeros(50), values_mapping=mapping)
        dual.array[3:10] = 'Dual'

        mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}
        pilot = M('Pilot Flying', np.ma.zeros(50), values_mapping=mapping)
        pilot.array[0:20] = 'Captain'

        takeoff_roll = buildsection('Takeoff Roll', 0, 5)
        landing_roll = buildsection('Landing Roll', 44, 50)

        node = self.node_class()
        node.derive(dual, pilot, takeoff_roll, landing_roll)

        name = self.node_class.get_name()
        expected = KPV(name=name, items=[
            KeyPointValue(name=name, index=3, value=7.0),
        ])

        self.assertEqual(node, expected)

    # FIXME: after the changes in the algorithm the following test does not
    # work (algorithm does not detect dual input)
    ####def test_derive_from_hdf(self):
    ####    path = os.path.join(test_data_path, 'dual_input.hdf5')
    ####    [dual, pilot], phase = self.get_params_from_hdf(path,
    ####        ['Dual Input', 'Pilot Flying'])

    ####    takeoff_roll = buildsection('Takeoff Roll', 0, 100)
    ####    landing_roll = buildsection('Landing Roll', 320, 420)

    ####    node = self.node_class()
    ####    node.derive(dual, pilot, takeoff_roll, landing_roll)

    ####    name = self.node_class.get_name()
    ####    expected = KPV(name=name, items=[
    ####        KeyPointValue(name=name, index=91, value=31.0),
    ####        KeyPointValue(name=name, index=213, value=59.0),
    ####    ])

    ####    self.assertEqual(node, expected)

class TestDualInputByCaptMax(unittest.TestCase):

    def setUp(self):
        ranges = []
        start = 0
        for x in range(2, 20, 2):
            ranges.append(np.ma.arange(start, x, 0.5))
            ranges.append(np.ma.arange(x, x-1, -0.5))
            start = x-1
        self.stick_array = np.ma.concatenate(ranges)

        self.node_class = DualInputByCaptMax

    def test_can_operate(self):
        expected = [('Sidestick Angle (Capt)',
                     'Dual Input',
                    'Pilot Flying',
                    'Takeoff Roll',
                    'Landing Roll')]
        self.assertEqual(expected,
                         self.node_class.get_operational_combinations())

    def test_derive(self):
        takeoff_roll = buildsection('Takeoff Roll', 5, 20)
        landing_roll = buildsection('Landing Roll', 55, 65)
        dual_inputs_array = np.ma.zeros(70)
        dual_inputs_array[25:30] = 1
        dual_inputs_array[66:70] = 1
        dual_inputs = M('Dual Input',
                        array=dual_inputs_array,
                        values_mapping={0: '-', 1: 'Dual'})
        pilot_array = np.ma.zeros(70)
        pilot_array[10:30] = 2
        pilot_array[40:50] = 2
        pilot_array[66:70] = 2
        pilot = M('Pilot Flying',
                  array=pilot_array,
                        values_mapping={0: '-', 1: 'Captain', 2: 'First Officer'})
        stick = P('Sidestick Angle (Capt)', array=self.stick_array)
        node = self.node_class()
        node.derive(stick, dual_inputs, pilot, takeoff_roll, landing_roll)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 28)
        self.assertEqual(node[0].value, 8)


class TestDualInputByFOMax(unittest.TestCase, NodeTest):

    def setUp(self):
        ranges = []
        start = 0
        for x in range(2, 20, 2):
            ranges.append(np.ma.arange(start, x, 0.5))
            ranges.append(np.ma.arange(x, x-1, -0.5))
            start = x-1
        self.stick_array = np.ma.concatenate(ranges)

        self.node_class = DualInputByFOMax

    def test_can_operate(self):
        expected = [('Sidestick Angle (FO)',
                     'Dual Input',
                    'Pilot Flying',
                    'Takeoff Roll',
                    'Landing Roll')]
        self.assertEqual(expected,
                         self.node_class.get_operational_combinations())

    def test_derive(self):
        takeoff_roll = buildsection('Takeoff Roll', 5, 20)
        landing_roll = buildsection('Landing Roll', 55, 65)
        dual_inputs_array = np.ma.zeros(70)
        dual_inputs_array[25:30] = 1
        dual_inputs_array[66:70] = 1
        dual_inputs = M('Dual Input',
                        array=dual_inputs_array,
                        values_mapping={0: '-', 1: 'Dual'})
        pilot_array = np.ma.zeros(70)
        pilot_array[10:30] = 1
        pilot_array[40:50] = 1
        pilot_array[66:70] = 1
        pilot = M('Pilot Flying',
                  array=pilot_array,
                        values_mapping={0: '-', 1: 'Captain', 2: 'First Officer'})
        stick = P('Sidestick Angle (FO)', array=self.stick_array)
        node = self.node_class()
        node.derive(stick, dual_inputs, pilot, takeoff_roll, landing_roll)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 28)
        self.assertEqual(node[0].value, 8)


##############################################################################


##############################################################################


class TestHoldingDuration(unittest.TestCase):
    # TODO: CreateKPVsFromSliceDurations test superclass.
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestTwoDegPitchTo35FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TwoDegPitchTo35FtDuration
        self.operational_combinations = [('2 Deg Pitch To 35 Ft',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLastFlapChangeToTakeoffRollEndDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LastFlapChangeToTakeoffRollEndDuration
        self.operational_combinations = [
            ('Flap Lever', 'Takeoff Roll Or Rejected Takeoff'),
            ('Flap Lever (Synthetic)', 'Takeoff Roll Or Rejected Takeoff'),
            ('Flap Lever', 'Flap Lever (Synthetic)',
             'Takeoff Roll Or Rejected Takeoff'),
        ]

    def test_derive(self):
        flap_array = np.ma.array([15, 15, 20, 20, 15, 15])
        flap_lever = M(
            name='Flap Lever', array=flap_array,
            values_mapping={f: str(f) for f in np.ma.unique(flap_array)},
        )
        takeoff_roll = S(items=[Section('Takeoff Roll', slice(0, 5), 0, 5)])
        node = self.node_class()
        node.derive(flap_lever, None, takeoff_roll)
        expected = [
            KeyPointValue(
                index=3.5, value=1.5,
                name='Last Flap Change To Takeoff Roll End Duration')
        ]
        self.assertEqual(list(node), expected)
        flap_synth = M(
            name='Flap Lever (Synthetic)',
            array=np.ma.array([15, 15, 20, 15, 15, 15]),
            values_mapping={f: str(f) for f in np.ma.unique(flap_array)},
        )
        node = self.node_class()
        node.derive(None, flap_synth, takeoff_roll)
        expected = [
            KeyPointValue(
                index=2.5, value=2.5,
                name='Last Flap Change To Takeoff Roll End Duration')
        ]
        self.assertEqual(list(node), expected)


class TestAirspeedMinusVMOMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVMOMax
        self.operational_combinations = [
            ('VMO', 'Airspeed', 'Airborne'),
            ('VMO Lookup', 'Airspeed', 'Airborne'),
        ]
        array = [300 + 40 * math.sin(n / (2 * math.pi)) for n in range(20)]
        self.airspeed = P('Airspeed', np.ma.array(array))
        self.vmo_record = P('VMO', np.ma.repeat(330, 20))
        self.vmo_lookup = P('VMO Lookup', np.ma.repeat(335, 20))
        self.airborne = buildsection('Airborne', 5, 15)

    def test_derive__record_only(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, self.vmo_record, None, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=9.991386482538246, name=name),
        ]))

    def test_derive__lookup_only(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, None, self.vmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=4.991386482538246, name=name),
        ]))

    def test_derive__prefer_record(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, self.vmo_record, self.vmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=9.991386482538246, name=name),
        ]))

    def test_derive__record_masked(self):
        self.vmo_record.array.mask = True
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, self.vmo_record, self.vmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=4.991386482538246, name=name),
        ]))

    def test_derive__both_masked(self):
        self.vmo_record.array.mask = True
        self.vmo_lookup.array.mask = True
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, self.vmo_record, self.vmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive__masked_within_phase(self):
        self.vmo_record.array[:-1] = np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.airspeed, self.vmo_record, self.vmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=4.991386482538246, name=name),
        ]))


class TestMachMinusMMOMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachMinusMMOMax
        self.operational_combinations = [
            ('MMO', 'Mach', 'Airborne'),
            ('MMO Lookup', 'Mach', 'Airborne'),
        ]
        array = [0.8 + 0.04 * math.sin(n / (2 * math.pi)) for n in range(20)]
        self.mach = P('Mach', np.ma.array(array))
        self.mmo_record = P('MMO', np.ma.repeat(0.83, 20))
        self.mmo_lookup = P('MMO Lookup', np.ma.repeat(0.82, 20))
        self.airborne = buildsection('Airborne', 5, 15)

    def test_derive__record_only(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, self.mmo_record, None, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.009991386482538389, name=name),
        ]))

    def test_derive__lookup_only(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, None, self.mmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.019991386482538398, name=name),
        ]))

    def test_derive__prefer_record(self):
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, self.mmo_record, self.mmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.009991386482538389, name=name),
        ]))

    def test_derive__record_masked(self):
        self.mmo_record.array.mask = True
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, self.mmo_record, self.mmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.019991386482538398, name=name),
        ]))

    def test_derive__both_masked(self):
        self.mmo_record.array.mask = True
        self.mmo_lookup.array.mask = True
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, self.mmo_record, self.mmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[]))

    def test_derive__masked_within_phase(self):
        self.mmo_record.array[:-1] = np.ma.masked
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.mach, self.mmo_record, self.mmo_lookup, self.airborne)
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=10, value=0.019991386482538398, name=name),
        ]))


########################################
# Aircraft Energy


class TestKineticEnergyAtRunwayTurnoff(unittest.TestCase):
    def test_derive(self):

        turn_off = KTI('Landing Turn Off Runway',
                       items=[KeyTimeInstance(10, 'Landing Turn Off Runway')])

        array = np.ma.array([0, 10, 20, 30, 40, 40, 40, 40, 30, 20, 10, 0])
        kinetic_energy = P('Kinetic Energy', array=array )

        ke = KineticEnergyAtRunwayTurnoff()
        ke.derive(turn_off, kinetic_energy)

        self.assertEqual( ke[0].value, 10 )


class TestAircraftEnergyWhenDescending(unittest.TestCase):
    def test_derive(self):
        aircraft_energy_array = np.ma.arange(20000, 1000, -100)
        aircraft_energy = P('Aircraft Energy',
                            array=aircraft_energy_array)

        descending = KTI('Altitude When Descending',
                         items =[KeyTimeInstance(100, '10000 Ft Descending'),
                                 KeyTimeInstance(150.5, '5000 Ft Descending')])

        aewd = AircraftEnergyWhenDescending()
        aewd.derive(aircraft_energy, descending)

        self.assertEqual(aewd[0].value, 10000)

        # value_at_index = takes the linear interpolation value at the given index
        # value_at_index is used in the derive method
        # Altitude at index 150 = 5000
        # Altitude at index 151 = 4900
        # Altitude at index 150.5 = 4950 (interpolated)
        self.assertEqual(aewd[1].value, 4950)

        self.assertEqual(aewd[0].name, 'Aircraft Energy at 10000 Ft Descending')
        self.assertEqual(aewd[1].name, 'Aircraft Energy at 5000 Ft Descending')


class TestAileronPreflightCheck(unittest.TestCase):

    def setUp(self):
        self.node_class = AileronPreflightCheck
        self.model = A('Model', 'Model')
        self.series = A('Series', 'Series')
        self.family = A('Family', 'Family')
        self.return_value = (-17, 25)

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, at):
        at.get_aileron_range.return_value = self.return_value
        aileron_deps = ['Aileron', 'Aileron (L)', 'Aileron (R)']
        start_deps = ['First Eng Start Before Liftoff', 'Last APU Start Before Liftoff']
        required_deps = ['Takeoff Acceleration Start', 'Model', 'Series', 'Family']

        # Empty tuple omitted from powerset
        start_deps_powerset = list(itertools.chain.from_iterable(itertools.combinations(start_deps, r)
                                                                 for r in range(len(start_deps) + 1)))[1:]
        aileron_deps_powerset = list(itertools.chain.from_iterable(itertools.combinations(aileron_deps, r)
                                                                   for r in range(len(aileron_deps) + 1)))[1:]

        ops = self.node_class.get_operational_combinations(model=self.model, series=self.series, family=self.family)

        for i in range(len(start_deps_powerset)):
            for j in range(len(aileron_deps_powerset)):
                self.assertIn((tuple(aileron_deps_powerset[j]) + tuple(start_deps_powerset[i]) + tuple(required_deps)), ops)

        at.get_aileron_range.side_effect = KeyError('No Aileron range for model')
        self.assertEqual(self.node_class.get_operational_combinations(model=self.model, series=self.series, family=self.family),
                         [])

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, at):
        firsts = KTI('First Eng Start Before Liftoff',
                     items=[KeyTimeInstance(50, 'First Eng Start Before Liftoff')])
        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        aileron = P(name='Aileron', array=x*np.sin(x)*3)

        # Assume that lookup tables are found correctly...
        at.get_aileron_range.return_value = self.return_value

        for args in itertools.product(*[(None, aileron)] * 3):
            if not any(args):
                continue
            node = self.node_class()
            node.derive(*list(args) + [firsts, None, accels, self.model, self.series, self.family])
            self.assertEqual(len(node), 1)
            self.assertEqual(node[0].index, 318)
            self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement

    @patch('analysis_engine.key_point_values.at')
    def test_derive_apu(self, at):
        lasts = KTI('Last APU Start Before Liftoff',
                    items=[KeyTimeInstance(50, 'Last APU Start Before Liftoff')])
        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        aileron = P(name='Aileron', array=x*np.sin(x)*3)

        # Assume that lookup tables are found correctly...
        at.get_aileron_range.return_value = self.return_value

        for args in itertools.product(*[(None, aileron)] * 3):
            if not any(args):
                continue
            node = self.node_class()
            node.derive(*list(args) + [None, lasts, accels, self.model, self.series, self.family])
            self.assertEqual(len(node), 1)
            self.assertEqual(node[0].index, 318)
            self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement


class TestElevatorPreflightCheck(unittest.TestCase):

    def setUp(self):
        self.node_class = ElevatorPreflightCheck
        self.model = A('Model', 'Model')
        self.series = A('Series', 'Series')
        self.family = A('Family', 'Family')
        self.return_value = (-17, 25)

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, at):
        at.get_aileron_range.return_value = self.return_value
        elevator_deps = ['Elevator', 'Elevator (L)', 'Elevator (R)']
        start_deps = ['First Eng Start Before Liftoff', 'Last APU Start Before Liftoff']
        required_deps = ['Takeoff Acceleration Start', 'Model', 'Series', 'Family']

        # Empty tuple omitted from powerset
        start_deps_powerset = list(itertools.chain.from_iterable(itertools.combinations(start_deps, r)
                                                                 for r in range(len(start_deps) + 1)))[1:]
        elevator_deps_powerset = list(itertools.chain.from_iterable(itertools.combinations(elevator_deps, r)
                                                                    for r in range(len(elevator_deps) + 1)))[1:]

        ops = self.node_class.get_operational_combinations(model=self.model, series=self.series, family=self.family)

        for i in range(len(start_deps_powerset)):
            for j in range(len(elevator_deps_powerset)):
                self.assertIn((tuple(elevator_deps_powerset[j]) + tuple(start_deps_powerset[i]) + tuple(required_deps)), ops)

        at.get_elevator_range.side_effect = KeyError('No Elevator range for model')
        self.assertEqual(self.node_class.get_operational_combinations(model=self.model, series=self.series, family=self.family),
                         [])

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, at):
        firsts = KTI('First Eng Start Before Liftoff',
                     items=[KeyTimeInstance(50, 'First Eng Start Before Liftoff')])
        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        elevator = P(name='Elevator', array=x*np.sin(x)*3)

        # Assume that lookup tables are found correctly...
        at.get_elevator_range.return_value = self.return_value

        for args in itertools.product(*[(None, elevator)] * 3):
            if not any(args):
                continue
            node = self.node_class()
            node.derive(*list(args) + [firsts, None, accels, self.model, self.series, self.family])
            self.assertEqual(len(node), 1)
            self.assertEqual(node[0].index, 318)
            self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement

    @patch('analysis_engine.key_point_values.at')
    def test_derive_apu(self, at):
        lasts = KTI('Last APU Start Before Liftoff',
                    items=[KeyTimeInstance(50, 'Last APU Start Before Liftoff')])
        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        elevator = P(name='Elevator', array=x*np.sin(x)*3)

        # Assume that lookup tables are found correctly...
        at.get_elevator_range.return_value = self.return_value

        for args in itertools.product(*[(None, elevator)] * 3):
            if not any(args):
                continue
            node = self.node_class()
            node.derive(*list(args) + [None, lasts, accels, self.model, self.series, self.family])
            self.assertEqual(len(node), 1)
            self.assertEqual(node[0].index, 318)
            self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement

    @patch('analysis_engine.key_point_values.at')
    def test_derive_eng_before_apu(self, at):
        elev_l = P(name='Elevator (L)', array=load_compressed(os.path.join(test_data_path, 'ElevatorPreflightCheck_elev_l.npz')))
        elev_r = P(name='Elevator (R)', array=load_compressed(os.path.join(test_data_path, 'ElevatorPreflightCheck_elev_r.npz')))
        eng_firsts = KTI('First Eng Start Before Liftoff',
                         items=[KeyTimeInstance(index=8525.3671875, name='First Eng Start Before Liftoff')])
        apu_lasts = KTI('Last APU Start Before Liftoff',
                        items=[KeyTimeInstance(10905.9453125, 'Last APU Start Before Liftoff')])
        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(11133.4921875, 'Takeoff Acceleration Start')])
        at.get_elevator_range.return_value = (-30, 15)

        node = self.node_class()
        node.derive(None, elev_l, elev_r, eng_firsts, apu_lasts, accels, A('Model', 'A340-313X'), A('Series', 'A340-300'),
                    A('Family', 'A340'))
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 8992, places=0)
        self.assertAlmostEqual(node[0].value, 98, places=0)


class TestRudderPreflightCheck(unittest.TestCase):

    def setUp(self):
        self.node_class = RudderPreflightCheck
        self.model = A('Model', 'Model')
        self.series = A('Series', 'Series')
        self.family = A('Family', 'Family')
        self.return_value = (-17, 25)

    @patch('analysis_engine.key_point_values.at')
    def test_can_operate(self, at):
        required_deps = ['Takeoff Acceleration Start', 'Model', 'Series', 'Family']

        ops = self.node_class.get_operational_combinations(model=self.model, series=self.series, family=self.family)
        # assertIn didn't work
        self.assertTrue(tuple(['Rudder', 'First Eng Start Before Liftoff'] + required_deps) in ops or
                        (tuple(['Rudder', 'Last APU Start Before Liftoff'] + required_deps) in ops))

    @patch('analysis_engine.key_point_values.at')
    def test_derive(self, at):
        firsts = KTI('First Eng Start Before Liftoff',
                     items=[KeyTimeInstance(50, 'First Eng Start Before Liftoff')])

        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        rudder = P(
            name='Rudder',
            array=x*np.sin(x)*3,
        )

        # Assume that lookup tables are found correctly...
        at.get_rudder_range.return_value = self.return_value

        node = self.node_class()
        node.derive(rudder, firsts, None, accels, self.model, self.series, self.family)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 318)
        self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement

    @patch('analysis_engine.key_point_values.at')
    def test_derive_apu(self, at):
        lasts = KTI('Last APU Start Before Liftoff',
                    items=[KeyTimeInstance(50, 'Last APU Start Before Liftoff')])

        accels = KTI('Takeoff Acceleration Start',
                     items=[KeyTimeInstance(375, 'Takeoff Acceleration Start')])
        x = np.linspace(0, 10, 400)
        rudder = P(
            name='Rudder',
            array=x*np.sin(x)*3,
        )

        # Assume that lookup tables are found correctly...
        at.get_rudder_range.return_value = self.return_value

        node = self.node_class()
        node.derive(rudder, None, lasts, accels, self.model, self.series, self.family)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 318)
        self.assertAlmostEqual(node[0].value, 90, delta=1) # 90% of total movement


class TestFlightControlPreflightCheck(unittest.TestCase):

    def setUp(self):
        self.node_class = FlightControlPreflightCheck

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         [('Elevator Preflight Check', 'Aileron Preflight Check', 'Rudder Preflight Check')])

    def test_derive(self):
        ele = KPV(name='Elevator Preflight Check', items=[
            KeyPointValue(index=7.0, value=100, name='Elevator Preflight Check'),
        ])
        ail = KPV(name='Aileron Preflight Check', items=[
            KeyPointValue(index=9.0, value=103, name='Aileron Preflight Check'),
        ])
        rud = KPV(name='Rudder Preflight Check', items=[
            KeyPointValue(index=14.0, value=97, name='Rudder Preflight Check'),
        ])

        node = self.node_class()
        node.derive(ele, ail, rud)

        name = 'Flight Control Preflight Check'
        self.assertEqual(node, KPV(name=name, items=[
            KeyPointValue(index=7.0, value=300, name=name),
        ]))


class TestDriftAtTouchdown(unittest.TestCase):

    def setUp(self):
        self.node_class = DriftAtTouchdown

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Drift At Touchdown')
        self.assertEqual(node.units, ut.DEGREE)

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 2)
        self.assertIn('Drift', opts[0])
        self.assertIn('Touchdown', opts[0])

    def test_derive(self):
        drift = P('Drift', np.ma.array([
            -9.9, -9.7, -9.1, -8.7, -9.0, -9.3, -9.4, -8.9, -8.3, -8.2,
            -6.5, -5.5, -4.9, -4.7, -2.9, -0.4, -1.2, -2.3, -2.3, -2.8,
            -2.1, -2.0, -2.5, -2.5, -2.5, -2.1, -1.7, -1.9, -1.8, -2.5
        ]))
        touchdown = KTI(items=[KeyTimeInstance(index=13, name='Touchdown'),])

        node = self.node_class()
        node.derive(drift, touchdown)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, -4.7)


class TestDHSelectedAt1500FtLVO(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = DHSelectedAt1500FtLVO

        self.alt_std_desc = AltitudeWhenDescending(
            items=[KeyTimeInstance(15, '1500 Ft Descending'),
                   KeyTimeInstance(20, '1500 Ft Descending'),
                   KeyTimeInstance(28, '1500 Ft Descending'),
                   KeyTimeInstance(44, '1500 Ft Descending')])

        self.dh_selected = P('DH Selected', np.ma.concatenate(
                            (np.zeros(25),
                            np.ones(20) * 100,
                            np.zeros(25),
                            np.ones(20) * 100)))

        self.series_7x = A('Series', 'Falcon-7X')
        self.series_8x = A('Series', 'Falcon-8X')
        self.series_invalid = A('Series', 'Boeing B737-800')

    def test_can_operate(self):
        req_params = ('Altitude When Descending', 'DH Selected')

        self.assertFalse(self.node_class.can_operate(
                         req_params, series=self.series_invalid),
                         msg='KPV only applies to Falcon 7X/8X')

        self.assertTrue(self.node_class.can_operate(
                        req_params, series=self.series_7x))

        self.assertTrue(self.node_class.can_operate(
                        req_params, series=self.series_8x))

    def test_derive(self):
        node = self.node_class()
        node.derive(self.alt_std_desc, self.dh_selected)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 28)
        self.assertEqual(node[0].value, 100.0)
        self.assertEqual(node[1].index, 44)
        self.assertEqual(node[1].value, 100.0)


class TestTransmitInactivityDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TransmitInactivityDuration
        self.operational_combinations = [('Transmitting', 'Airborne')]

    def test_derive(self):
        transmitting = M('Transmitting', array=MappedArray(
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            values_mapping={0: '-', 1: 'Transmit'},
        ))
        airs = buildsection('Airborne', 1, 12)

        node = self.node_class()
        node.derive(transmitting, airs)

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 7)
        self.assertAlmostEqual(node[0].value, 3)


if __name__ == '__main__':
    unittest.main()
