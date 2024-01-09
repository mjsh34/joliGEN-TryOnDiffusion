import argparse
from pathlib import Path
import zipfile
from tqdm import tqdm
import os
import shutil
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("VITON-HD dataset preprocessing")
    parser.add_argument(
        "--zip-file",
        type=str,
        required=True,
        help="path to the VITON-HD zip file",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="path to an output folder for preprocessed data",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=1,
        help="size of the square kernel for mask dilation",
    )
    parser.add_argument(
        "--save_conditions",
        action='store_true')
    parser.add_argument('--padding', default="", help="{top_padding} {left_padding} {bottom_padding} {right_padding}")
    parser.add_argument(
        "--bbox_fill_image",
        action='store_true',
        help="If set, the bbox will span the entire image")
    parser.add_argument(
        "--bbox_fill_image__cloth",
        action='store_true',
        help="Only active when --bbox_fill_image is set. If enabled, bbox will span the 'cloth' images in the VITON-HD "
        "dataset.")
    parser.add_argument(
        "--debug_vis_bbox",
        action='store_true',
        help="Model shots inside imgs/ will have bounding box superimposed. For debugging purposes only.")
    return parser.parse_args()


def process(image, zf, target_dir, dilate, save_conditions=False, **kw):
    stage = Path("trainA" if "train/" in image else "testA")
    basename = Path(image).stem

    padding = kw.get("padding") or {
            "top": 5,
            "bottom": 15,
            "left": 20,
            "right": 20,
    }

    # extract raw image
    rel_image = stage / "imgs" / (basename + ".jpg")
    target_img = target_dir / rel_image
    target_img.write_bytes(zf.read(image))

    # extract mask
    mask = image.replace("/image/", "/image-parse-v3/").replace(".jpg", ".png")
    mask = zf.read(mask)
    mask = cv2.imdecode(np.frombuffer(mask, np.uint8), 1)
    orange = np.array([0, 85, 254])
    mask = cv2.inRange(mask, orange, orange)
    mask = 255 * np.clip(mask, 0, 1)
    kernel = np.ones((dilate, dilate), np.uint8)
    mask = cv2.dilate(mask, kernel)
    rel_mask = stage / "mask" / (basename + ".png")
    target_mask = target_dir / rel_mask
    cv2.imwrite(str(target_mask), mask)

    if save_conditions:
        # extract bbox
        mask = image.replace("/image/", "/image-parse-v3/").replace(".jpg", ".png")
        mask = zf.read(mask)
        mask = cv2.imdecode(np.frombuffer(mask, np.uint8), 1)
        orange = np.array([0, 85, 254])
        mask = cv2.inRange(mask, orange, orange)
        mask = np.clip(mask, 0, 1)
        masked_inds = np.nonzero(mask > 0)
        if not kw.get('bbox_fill_image', False):
            try:
                top = masked_inds[0].min()
                bottom = masked_inds[0].max()
                left = masked_inds[1].min()
                right = masked_inds[1].max()
                mask_w, mask_h = right - left, bottom - top
                assert mask_w > 0 and mask_h > 0
            except:
                print("Cannot identify proper mask for '{}'. Skipping...".format(image))
                os.remove(target_img)
                os.remove(target_mask)
                return False

            top = max(0, top - padding['top'])
            left = max(0, left - padding['left'])
            bottom = min(mask.shape[0] - 1, bottom + padding['bottom'])
            right = min(mask.shape[1] - 1, right + padding['right'])
        else: # bbox_fill_image
            if kw.get('bbox_fill_images__cloth', False):
                cloth = image.replace("/image/", "/cloth/")
                cloth = zf.read(cloth)
                cloth = cv2.imdecode(np.frombuffer(cloth, np.uint8), 1)
                top, left, bottom, right = 0, 0, cloth.shape[0] - 1, cloth.shape[1] - 1
            else:
                top, left, bottom, right = 0, 0, mask.shape[0] - 1, mask.shape[1] - 1


        if kw.get('debug_vis_bbox', False):
            target_im = cv2.imread(str(target_img)) #cv2.imdecode(np.frombuffer(zf.read(image), np.uint8), 1)
            target_im[top:bottom+1, left:right+1] = (0.5 * target_im[top:bottom+1, left:right+1]) + 0.5
            cv2.imwrite(str(target_img), target_im)

        # bbox
        rel_bbox = stage / "bbox" / (basename + ".txt")
        target = target_dir / rel_bbox
        target.open('w').write("1 {} {} {} {}\n".format(left, top, right, bottom))

        # ref
        rel_ref = stage / "ref" / (basename + ".jpg")
        garment = zf.read(image.replace("/image/", "/cloth/"))
        garment = cv2.imdecode(np.frombuffer(garment, np.uint8), 1)
        target = target_dir / rel_ref
        cv2.imwrite(str(target), garment)

        # conditions
        rel_cond = stage / "cond" / (basename + ".txt")
        target = target_dir / rel_cond
        target.open('w').write(str(stage / "ref" / (basename + ".jpg")) + "\n")

        # append to conditions.txt
        pairs = target_dir / stage / "conditions.txt"
        pairs = pairs.open("a")
        pairs.write(f"{rel_image} {rel_cond}\n")

        # add paths
        pairs = target_dir / stage / "paths.txt"
        pairs = pairs.open("a")
        pairs.write(f"{rel_image} {rel_bbox}\n")
    else:
        # add paths
        pairs = target_dir / stage / "paths.txt"
        pairs = pairs.open("a")
        pairs.write(f"{rel_image} {rel_mask}\n")

    return True


def main():
    args = parse_args()

    padding = None
    if args.padding:
        padding_tlbr = [int(p) for p in args.padding.split(" ")]
        padding = {
            'top': padding_tlbr[0],
            'left': padding_tlbr[1],
            'bottom': padding_tlbr[2],
            'right': padding_tlbr[3]
        }

    # create dataset folders
    zip_file = Path(args.zip_file)
    assert zip_file.is_file()
    target_dir = Path(args.target_dir)
    #assert not target_dir.exists()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    for folder1 in ["trainA", "testA"]:
        for folder2 in ["imgs", "mask", "bbox", "ref", "cond"]:
            folder = target_dir / folder1 / folder2
            folder.mkdir(parents=True)

    # process images
    zf = zipfile.ZipFile(zip_file)
    images = [name for name in zf.namelist() if "/image/" in name and "_00.jpg" in name]
    nfailed = 0
    for image in tqdm(images):
        success = process(image, zf, target_dir, args.dilate,
                save_conditions=args.save_conditions, bbox_fill_image=args.bbox_fill_image, debug_vis_bbox=args.debug_vis_bbox,
                bbox_fill_image__cloth=args.bbox_fill_image__cloth,
                padding=padding)
        if not success:
            nfailed += 1

    print("Processed {} images; {} succeded ({} failed).".format(
        len(images), len(images) - nfailed, nfailed))


if __name__ == "__main__":
    main()
