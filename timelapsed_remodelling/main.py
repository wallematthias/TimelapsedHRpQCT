
from glob import glob
import os
import argparse
from timelapsed_remodelling import custom_logger
from timelapsed_remodelling.timelapse import TimelapsedImageSeries
from timelapsed_remodelling.register import Registration
import re

def get_sort_key(filename):
    # Extract the part after the last underscore and sort based on that
    return filename.split('_')[-1]

def process_patients(paths, result_pairs, tibia_identifiers, resolution, output_path, trabmask, cortmask, threshold=225, cluster=12):
    
    patient_name = os.path.basename(paths[0])

    if any(string in patient_name for string in tibia_identifiers):
        site = 'tibia'
    else:
        site = 'radius'

    # Create an instance of the registration
    reg = Registration(
        sampling=0.01,
        num_of_iterations=100)
    reg.setOptimizer('powell')
    reg.setMultiResolution(
        shrinkFactors=[12, 8, 4, 2, 1, 1],
        smoothingSigmas=[0, 0, 0, 0, 1, 0])
    reg.setInterpolator('linear')
    reg.setSimilarityMetric('correlation')
    processor = TimelapsedImageSeries(site, patient_name, resolution=resolution, crop=True)

    for i, scan in enumerate(paths):
        custom_logger.info(scan)
        processor.add_image(str(i), scan)
        
        clean_scan_name = re.sub(r';\d+', '', scan)
        custom_logger.info(clean_scan_name)
        trab_mask_path = glob(clean_scan_name.replace('.AIM',f'_{trabmask}.AIM*'))
        cort_mask_path = glob(clean_scan_name.replace('.AIM',f'_{cortmask}.AIM*'))
        custom_logger.info(trab_mask_path[0])
        custom_logger.info(cort_mask_path[0])
        if len(trab_mask_path+cort_mask_path)>0:
            processor.add_contour_to_image(str(i),trabmask,trab_mask_path[0])
            processor.add_contour_to_image(str(i),cortmask,cort_mask_path[0])
        else: #generate if not given                                                
            processor.generate_contour(str(i),path=output_path)
    processor.debug(outpath=output_path)   
    #Activate motion grading
    processor.motion_grade(outpath=output_path)
    processor.register(reg,path=output_path)
    # Convert input into pairs of tuples (baseline, followup)
    pairs = [tuple(result_pairs[i:i+2]) for i in range(0, len(result_pairs), 2)]
    for i, (baseline, followup) in enumerate(pairs, 1):
        #try:
        processor.analyse(str(baseline), str(followup), threshold=threshold, cluster=cluster, outpath=output_path)
        #except Exception as e:
        #    print(e)
        #    custom_logger.info(f'Could not analyse baseline {baseline} followup {followup}')

    processor.save(str(result_pairs[0]), output_path,visualise=False)
    

def main():
    parser = argparse.ArgumentParser(description='Process patients from a directory using glob pattern.')
    parser.add_argument('paths', nargs="+", default='*.AIM*', type=str, help='Path to the directory containing the patient files.')
    parser.add_argument('--resolution', type=float, default=0.06069965288043022, help='Resolution value in mm.')
    parser.add_argument('--output_path', type=str, default='.', help='Path to the output directory.')
    parser.add_argument(
        "--result_pairs",
        nargs="+",
        type=str,
        default = ['0','1'],
        help="Baseline-followup pairs. Each pair should be separated by a space.")
    parser.add_argument(
        "--tibia_identifiers",
        nargs="+",
        type=str,
        default = ['DT','LT','RT','TR','TL'],
        help="Baseline-followup pairs. Each pair should be separated by a space.")
    
    parser.add_argument('--trabmask', default='TRAB_MASK')
    parser.add_argument('--cortmask', default='CORT_MASK')
    parser.add_argument('--cluster', type=float, default=12)
    parser.add_argument('--threshold', type=float, default=225)
    

    args = parser.parse_args()
    paths = [item for item in args.paths if "MASK" not in item]
    sorted_paths = sorted(paths)

    process_patients(sorted_paths, args.result_pairs, args.tibia_identifiers ,args.resolution, args.output_path, args.trabmask, args.cortmask, cluster=args.cluster, threshold=args.threshold)


if __name__ == "__main__":
    main()