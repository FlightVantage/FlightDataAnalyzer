import os
import logging
import numpy as np
import time

from hdfaccess.file import hdf_file
from hdfaccess.utils import write_segment

from analysis import settings
from analysis.plot_flight import plot_essential
from analysis.split_segments import append_segment_info, split_segments


class AircraftMismatch(ValueError):
    pass

    

##def deidentify_file(file_path):
    ##"""
    ##Removes any specific meta-data.
    ##Removes timebase / amends.
    ##Removes parameters.
    ##"""
    ##pass

def validate_aircraft(aircraft_ident, hdf):
    """
    """
    #if 'Aircraft Ident' in hdf and so on:
    logging.warning("Validate Aircraft not implemented")
    if True:
        return True
    else:
        raise AircraftMismatch("Tail does not match identification %s" % \
                               aircraft_ident['Tail Number'])
    
def post_lfl_param_process(hdf, param):
    if settings.POST_LFL_PARAM_PROCESS:
        # perform post lfl retrieval steps
        _param = settings.POST_LFL_PARAM_PROCESS(hdf, param)
        if _param:
            # store any updates to param to hdf file
            hdf.set_param(_param)
            return _param
    return param


def split_hdf_to_segments(hdf_path, aircraft_ident={}, output_dir=None, draw=False):
    """
    Main method - analyses an HDF file for flight segments and splits each
    flight into a new segment appropriately.
    
    :param hdf_path: path to HDF file
    :type hdf_path: string
    :param aircraft_ident: Information which identify the aircraft including 'Tail Number', 'MSN'...
    :type aircraft_ident: Dict
    :param output_dir: Directory to write the destination file to. If None, directory of source file is used.
    :type output_dir: String (path)
    :param draw: Whether to use matplotlib to plot the flight
    :type draw: Boolean
    :returns: List of Segments
    :rtype: List of Segment recordtypes ('slice type part duration path hash')
    """
    logging.info("Processing file: %s", hdf_path)
    if draw:
        plot_essential(hdf_path)
        
    with hdf_file(hdf_path) as hdf:
        if settings.PRE_FILE_ANALYSIS:
            logging.debug("Performing pre-file analysis: %s", settings.PRE_FILE_ANALYSIS.func_name)
            settings.PRE_FILE_ANALYSIS(hdf)
        
        # Confirm aircraft tail for the entire datafile
        if aircraft_ident:
            logging.info("Validating aircraft matches that recorded in data")
            validate_aircraft(aircraft_ident, hdf)
        else:
            logging.info("Not validating aircraft is correct")
        
        # uses flight phases and DFC if aircraft determines to do so
        airspeed = post_lfl_param_process(hdf, hdf['Airspeed'])
                
        # split large dataset into segments
        logging.debug("Splitting segments. Data length: %s", len(airspeed.array))
        if hdf.reliable_frame_counter:
            dfc = post_lfl_param_process(hdf, hdf['Frame Counter'])
        else:
            dfc = None
        segment_slices = split_segments(airspeed, dfc=dfc)
            
    # process each segment (into a new file) having closed original hdf_path
    segments = []
    for part, segment_slice in enumerate(segment_slices):
        part += 1 # one indexed part
        # write segment to new split file (.001)
        if output_dir:
            path = os.path.join(output_dir, os.path.basename(hdf_path))
        else:
            path = hdf_path
        dest_path = path.rstrip('.hdf5') + '.%03d' % part + '.hdf5'
        logging.debug("Writing segment %d: %s", part, dest_path)
        dest_path = write_segment(hdf_path, segment_slice, dest_path)
        
        segment = append_segment_info(dest_path, segment_slice, part)
        segments.append(segment)
        if draw:
            plot_essential(dest_path)
            
    if draw:
        # show all figures together
        from matplotlib.pyplot import show
        show()
        #close('all') # closes all figures
         
    return segments
            
      
if __name__ == '__main__':
    import sys
    import pprint
    hdf_path = sys.argv[1]
    segs = split_hdf_to_segments(hdf_path, draw=False)    
    pprint.pprint(segs)
    ##os.remove(file_path) # delete original raw data file?
    ##os.remove(hdf_path) # delete original hdf file?
    