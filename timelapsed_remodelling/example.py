
from glob import glob
import os
import argparse

from timelapsed_remodelling import custom_logger
from timelapsed_remodelling.timelapse import TimelapsedImageSeries
from timelapsed_remodelling.register import Registration

def get_sort_key(filename):
    # Extract the part after the last underscore and sort based on that
    return filename.split('_')[-1]

def process_patients(paths, result_pairs, tibia_identifiers, resolution, output_path):
    
    patient_name = os.path.basename(paths[0])

    if any(string in patient_name for string in tibia_identifiers):
        site = 'tibia'
    else:
        site = 'radius'
    print('here')
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
    print('here2')
    processor = TimelapsedImageSeries(site, patient_name, resolution=resolution, crop=True)

    for i, scan in enumerate(paths):
        custom_logger.info(scan)
        processor.add_image(str(i), scan)
        processor.generate_contour(str(i),path=output_path)
    print('here3')    
    processor.register(reg,path=output_path)
    # Convert input into pairs of tuples (baseline, followup)
    pairs = [tuple(result_pairs[i:i+2]) for i in range(0, len(result_pairs), 2)]
    for i, (baseline, followup) in enumerate(pairs, 1):
        custom_logger.info(baseline)
        custom_logger.info(followup)
        processor.analyse(str(baseline), str(followup), threshold=225, cluster=12)

    processor.save(str(result_pairs[0]), output_path)
    

def main():
    print('here0')
    parser = argparse.ArgumentParser(description='Process patients from a directory using glob pattern.')
    parser.add_argument('paths', nargs="+", default='*.AIM*', type=str, help='Path to the directory containing the patient files.')
    parser.add_argument('--resolution', type=float, default=0.06069965288043022, help='Resolution value in mm.')
    parser.add_argument('--output_path', type=str, default='.', help='Path to the output directory.')
    parser.add_argument(
        "--result_pairs",
        nargs="+",
        type=str,
        default = [['MOO', 'M06'],['M00','M12'],['M06','M12']],
        help="Baseline-followup pairs. Each pair should be separated by a space.")
    parser.add_argument(
        "--tibia_identifiers",
        nargs="+",
        type=str,
        default = ['DT','LT','RT','TR','TL'],
        help="Baseline-followup pairs. Each pair should be separated by a space.")

    args = parser.parse_args()
    process_patients(args.paths, args.result_pairs, args.tibia_identifiers ,args.resolution, args.output_path)


if __name__ == "__main__":
    main()