import os
import re
import argparse
from glob import glob

from timelapsed_remodelling import custom_logger
from timelapsed_remodelling.timelapse import TimelapsedImageSeries
from timelapsed_remodelling.register import Registration
from multiprocessing import resource_tracker

# Unregister semaphore cleanup (not recommended for general use, but works here)
def fix_leaking_semaphores():
    # Only do this if you're sure no semaphores need tracking
    try:
        resource_tracker._CLEANUP_FUNCS['semaphore'] = lambda name: None
    except Exception as e:
        print(f"Failed to patch resource_tracker: {e}")

def get_sort_key(filename):
    """Helper to extract the sorting key from filename."""
    return filename.split('_')[-1]


def process_patients(
    paths,
    result_pairs,
    tibia_identifiers,
    resolution,
    output_path,
    trabmask,
    cortmask,
    threshold=225,
    cluster=12,
):
    patient_name = os.path.basename(paths[0])
    site = "tibia" if any(tag in patient_name for tag in tibia_identifiers) else "radius"

    reg = Registration(sampling=0.01, num_of_iterations=100)
    reg.setOptimizer("powell")
    reg.setMultiResolution(
        shrinkFactors=[12, 8, 4, 2, 1, 1],
        smoothingSigmas=[0, 0, 0, 0, 1, 0],
    )
    reg.setInterpolator("linear")
    reg.setSimilarityMetric("correlation")

    processor = TimelapsedImageSeries(site, patient_name, resolution=resolution, crop=True)

    for idx, image_path in enumerate(paths):
        custom_logger.info(image_path)
        processor.add_image(str(idx), image_path)

        clean_name = re.sub(r";\d+", "", image_path)
        custom_logger.info(clean_name)

        trab_mask_path = glob(clean_name.replace(".AIM", f"_{trabmask}.*"))
        cort_mask_path = glob(clean_name.replace(".AIM", f"_{cortmask}.*"))

        if trab_mask_path or cort_mask_path:
            custom_logger.info(trab_mask_path[0])
            custom_logger.info(cort_mask_path[0])
            processor.add_contour_to_image(str(idx), trabmask, trab_mask_path[0])
            processor.add_contour_to_image(str(idx), cortmask, cort_mask_path[0])
        else:
            processor.generate_contour(str(idx), path=output_path)

    processor.debug(outpath=output_path)
    processor.register(reg, path=output_path)

    # Convert to list of (baseline, followup) tuples
    pairs = [tuple(result_pairs[i:i + 2]) for i in range(0, len(result_pairs), 2)]
    for i, (baseline, followup) in enumerate(pairs, 1):
        custom_logger.info(f"Processing pair {i}: {baseline} - {followup}")
        processor.analyse(str(baseline), str(followup), threshold=threshold, cluster=cluster, outpath=output_path)

    custom_logger.info("Finished processing all pairs.")
    processor.save(str(result_pairs[0]), output_path, visualise=True)


def main():
    fix_leaking_semaphores()

    parser = argparse.ArgumentParser(
        description="Process patients from a directory using glob pattern."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        default="*.AIM*",
        help="Paths to the patient scan files."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.06069965288043022,
        help="Voxel resolution in mm."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Output directory."
    )
    parser.add_argument(
        "--result_pairs",
        nargs="+",
        type=str,
        default=["0", "1"],
        help="Pairs of indices specifying baseline and followup scans (e.g., 0 1 0 2)."
    )
    parser.add_argument(
        "--tibia_identifiers",
        nargs="+",
        type=str,
        default=["DT", "LT", "RT", "TR", "TL"],
        help="Keywords used to classify scans as tibia."
    )
    parser.add_argument("--trabmask", default="TRAB_MASK", help="Name identifier for trabecular mask.")
    parser.add_argument("--cortmask", default="CORT_MASK", help="Name identifier for cortical mask.")
    parser.add_argument("--cluster", type=float, default=12, help="Clustering parameter for remodeling analysis.")
    parser.add_argument("--threshold", type=float, default=225, help="Threshold for remodeling analysis.")

    args = parser.parse_args()

    input_paths = [p for p in args.paths if "MASK" not in p]
    sorted_paths = sorted(input_paths)

    process_patients(
        sorted_paths,
        args.result_pairs,
        args.tibia_identifiers,
        args.resolution,
        args.output_path,
        args.trabmask,
        args.cortmask,
        cluster=args.cluster,
        threshold=args.threshold,
    )
    custom_logger.info('Finished processing all patients.')
    fix_leaking_semaphores()
    os._exit(0)

if __name__ == "__main__":
    main()
