import argparse
from processes import apply_box_filter, apply_gaussian_filter, apply_shift_filter, apply_laplace_filter, apply_sobels_filter, sharpen_img_laplace, sharpen_img_unsharpening_mask, emboss_image
import imageio.v3 as iio
import numpy as np
from videomaker import create_video_fft

def luminance(pixel):
    return (0.2126*pixel[0] + 0.7152*pixel[1] + 0.0722*pixel[2])

def parse_args():
    parser = argparse.ArgumentParser(description='Image processor')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('output_image', type=str, help='Path to the output image')
    parser.add_argument('--box', type=int, help='Apply box filter with given size')
    parser.add_argument('--gaussian', nargs=2, type=float, help='Apply Gaussian filter with given sigma and size')
    parser.add_argument('--shift', nargs=2, type=int, help='Apply shift filter with given shift_y and shift_x')
    parser.add_argument('--laplace', action='store_true', help='Apply Laplace filter')
    parser.add_argument('--sobel', action='store_true', help='Apply Sobel filter')
    parser.add_argument('--sharpen_laplace', action='store_true', help='Sharpen image using Laplace')
    parser.add_argument('--sharpen_unsharpening_mask', nargs=2, type=float, help='Sharpen image using unsharpening mask with given size and sigma')
    parser.add_argument('--emboss', type=float, help='Apply emboss filter')
    parser.add_argument('--create_video_fft', action='store_true', help='Create video from FFT filters')

    return parser.parse_args()

def main():
    args = parse_args()
    raw_image = iio.imread(args.input_image).astype(np.float32)
    image = np.zeros((raw_image.shape[0], raw_image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = luminance(raw_image[i][j])
    
    if args.create_video_fft:
        create_video_fft(image, args.output_image)
        return
    if args.box:
        result = apply_box_filter(image, args.box)
    elif args.gaussian:
        sigma, size = args.gaussian
        result = apply_gaussian_filter(image, sigma, int(size))
    elif args.shift:
        shift_y, shift_x = args.shift
        result = apply_shift_filter(image, shift_y, shift_x)
    elif args.laplace:
        result = apply_laplace_filter(image)
    elif args.sobel:
        grad_x, grad_y = apply_sobels_filter(image)
        result = (grad_x, grad_y)  # Handle this case separately when saving
    elif args.sharpen_laplace:
        result = sharpen_img_laplace(image)
    elif args.sharpen_unsharpening_mask:
        size, sigma = args.sharpen_unsharpening_mask
        result = sharpen_img_unsharpening_mask(image, int(size), sigma)
    elif args.emboss:
        result = emboss_image(image, args.emboss)
    else:
        print("No valid filter specified.")
        return
    
    # Rescale result to clip to [0, 255] and convert to uint8
    if isinstance(result, tuple):
        result = tuple(np.clip(r, 0, 255).astype(np.uint8) for r in result)
    else:
        result = np.clip(result, 0, 255).astype(np.uint8)

    # Save result (placeholder, replace with actual image saving)
    # Replace with actual image saving code
    if isinstance(result, tuple):
        iio.imwrite(args.output_image.replace('.png', '_sobel_x.png'), result[0].astype(np.uint8))
        iio.imwrite(args.output_image.replace('.png', '_sobel_y.png'), result[1].astype(np.uint8))
    else:
        iio.imwrite(args.output_image, result.astype(np.uint8))
    iio.imwrite(args.output_image.replace('.png', '_original.png'), image.astype(np.uint8))


if __name__ == "__main__":
    main()
