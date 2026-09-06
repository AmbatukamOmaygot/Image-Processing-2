import argparse
import numpy as np
import skimage.io
import math


def mirror(img, axis='h'):
    """Отразить изображение"""
    h, w = img.shape
    res = np.zeros_like(img)
    
    if axis == 'v':
        for i in range(h):
            for j in range(w):
                res[i, j] = img[i, w - 1 - j]
    elif axis == 'h':
        for i in range(h):
            for j in range(w):
                res[i, j] = img[h - 1 - i, j]
    elif axis == 'd':
        res = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                res[j, i] = img[i, j]
    elif axis == 'cd':  
        res = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                res[w - 1 - j, h - 1 - i] = img[i, j]
    return res


def extract(img, left_x, top_y, width, height: int):
    """Извлечь фрагмент изображения"""
    res = np.zeros((height, width), dtype=img.dtype)
    
    for i in range(height):
        for j in range(width):
            src_i = top_y + i 
            src_j = left_x + j 
            
            if 0 <= src_i < img.shape[0] and 0 <= src_j < img.shape[1]:
                res[i, j] = img[src_i, src_j]
    
    return res


def rotate(img, direction, angle):
    """Поворот изображения на угол, кратный 90"""
    k = (angle // 90) % 4
    h, w = img.shape
    
    if direction == 'ccw':
        k = (4 - k) % 4
    
    if k == 0:
        res = img.copy()
    elif k == 1:
        res = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                res[j, h - 1 - i] = img[i, j]
    elif k == 2:
        res = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                res[h - 1 - i, w - 1 - j] = img[i, j]
    elif k == 3:
        res = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                res[w - 1 - j, i] = img[i, j]
    
    return res


def autocontrast(img):
    """Автоконтраст"""
    h, w = img.shape
    
    min_val = img[0, 0]
    max_val = img[0, 0]
    for i in range(h):
        for j in range(w):
            if img[i, j] < min_val:
                min_val = img[i, j]
            if img[i, j] > max_val:
                max_val = img[i, j]
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    res = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            res[i, j] = (img[i, j] - min_val) / (max_val - min_val)
    
    return res


def fixinterlace(img):
    """Обнаружение и коррекция чересстрочной развёртки"""
    h, w = img.shape
    
    variant1 = np.zeros_like(img)
    variant2 = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            if i % 2 == 0: 
                if i + 1 < h: 
                    variant1[i, j] = img[i + 1, j]
                else:
                    variant1[i, j] = img[i, j]
            else: 
                variant1[i, j] = img[i - 1, j]
    
    for i in range(h):
        for j in range(w):
            variant2[i, j] = img[i, j]
    
    def calc_variation(image):
        var = 0.0
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1]):
                var += abs(image[i + 1, j] - image[i, j])
        return var
    
    var1 = calc_variation(variant1)
    var2 = calc_variation(variant2)
    
    return variant1 if var1 < var2 else variant2

def get_pixel_with_border(img, i, j):
    """Дублирование ближайшего пикселя"""
    h, w = img.shape
    i_clamped = max(0, min(i, h - 1))
    j_clamped = max(0, min(j, w - 1))
    return img[i_clamped, j_clamped]


def median_filter(img, rad):
    """Медианная фильтрация"""
    h, w = img.shape
    res = np.zeros_like(img)
    window_size = 2 * rad + 1
    
    for i in range(h):
        for j in range(w):
            window = []
            for ki in range(-rad, rad + 1):
                for kj in range(-rad, rad + 1):
                    pixel = get_pixel_with_border(img, i + ki, j + kj)
                    window.append(pixel)

            window.sort()
            res[i, j] = window[len(window) // 2]
    
    return res


def gaussian_filter(img, sigma_d):
    """Фильтр Гаусса с параметром sigma_d"""
    h, w = img.shape
    res = np.zeros_like(img)
    radius = int(3 * sigma_d)

    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel_sum = 0.0
    
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            weight = math.exp(-(i*i + j*j) / (2 * sigma_d * sigma_d))
            kernel[i + radius, j + radius] = weight
            kernel_sum += weight

    kernel /= kernel_sum

    for i in range(h):
        for j in range(w):
            weighted_sum = 0.0
            for ki in range(-radius, radius + 1):
                for kj in range(-radius, radius + 1):
                    pixel = get_pixel_with_border(img, i + ki, j + kj)
                    weight = kernel[ki + radius, kj + radius]
                    weighted_sum += pixel * weight
            res[i, j] = weighted_sum
    
    return res


def bilateral(img, sigma_d, sigma_r):
    """Билатеральная фильтрация"""
    h, w = img.shape
    res = np.zeros_like(img)
    radius = int(3 * sigma_d)
    
    sigma_r_scaled = sigma_r / 255.0
    
    for i in range(h):
        for j in range(w):
            weighted_sum = 0.0
            weight_sum = 0.0
            center_pixel = img[i, j]
            
            for ki in range(-radius, radius + 1):
                for kj in range(-radius, radius + 1):
                    pixel = get_pixel_with_border(img, i + ki, j + kj)
                    
                    spatial_weight = math.exp(-(ki*ki + kj*kj) / (2 * sigma_d * sigma_d))
                    
                    intensity_diff = center_pixel - pixel
                    range_weight = math.exp(-(intensity_diff * intensity_diff) / (2 * sigma_r_scaled * sigma_r_scaled))
                    
                    total_weight = spatial_weight * range_weight
                    
                    weighted_sum += pixel * total_weight
                    weight_sum += total_weight
            
            if weight_sum > 0:
                res[i, j] = weighted_sum / weight_sum
            else:
                res[i, j] = img[i, j]
    
    return res

def mse(img1, img2):
    """MSE"""
    h, w = img1.shape
    total = 0.0
    
    for i in range(h):
        for j in range(w):
            err = (img1[i, j] * 255) - (img2[i, j] * 255)
            total += err * err
    
    return total / (h * w)


def psnr(img1, img2):
    """PSNR"""
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 20 * math.log10(255.0) - 10 * math.log10(mse_val)


def ssim(img1, img2):
    """SSIM"""
    h, w = img1.shape
    
    k1 = 0.01
    k2 = 0.03
    L = 255
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    mx = 0.0
    my = 0.0
    for i in range(h):
        for j in range(w):
            mx += img1[i, j] * L
            my += img2[i, j] * L
    mx /= (h * w)
    my /= (h * w)
    
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    
    for i in range(h):
        for j in range(w):
            dx = img1[i, j] * L - mx
            dy = img2[i, j] * L - my
            sx += dx * dx
            sy += dy * dy
            sxy += dx * dy
    
    sx /= (h * w)
    sy /= (h * w)
    sxy /= (h * w)
    
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sx + sy + c2)
    
    if den == 0:
        return 1.0
    
    return num / den


def blur(img, sigma):
    """Размытие"""
    h, w = img.shape
    res = np.zeros_like(img)
    radius = int(3 * sigma)
    
    if radius == 0:
        return img
    
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel_sum = 0.0
    
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            weight = math.exp(-(i*i + j*j) / (2 * sigma * sigma))
            kernel[i + radius, j + radius] = weight
            kernel_sum += weight
    
    kernel /= kernel_sum
    
    for i in range(h):
        for j in range(w):
            weighted_sum = 0.0
            for ki in range(-radius, radius + 1):
                for kj in range(-radius, radius + 1):
                    pixel = get_pixel_with_border(img, i + ki, j + kj)
                    weight = kernel[ki + radius, kj + radius]
                    weighted_sum += pixel * weight
            res[i, j] = weighted_sum
    
    return res

def compare_images(img1, img2):
    """Сравнение на предмет поворота или сдвига"""
    h, w = img1.shape
    
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    a1 = np.abs(f1)
    a2 = np.abs(f2)
    
    a1s = np.fft.fftshift(a1)
    a2s = np.fft.fftshift(a2)
    
    lb1 = blur(a1s, 1.0)
    lb2 = blur(a2s, 1.0)
    
    cy, cx = h // 2, w // 2
    r = min(cx, cy) // 2
    
    p1 = np.zeros((r, 360))
    p2 = np.zeros((r, 360))
    
    for ri in range(r):
        for t in range(360):
            tr = math.radians(t)
            x = cx + int(ri * math.cos(tr))
            y = cy + int(ri * math.sin(tr))
            
            if 0 <= x < w and 0 <= y < h:
                p1[ri, t] = lb1[y, x]
                p2[ri, t] = lb2[y, x]
    
    pr1 = np.concatenate([p1, p1[::-1]], axis=0)
    pr2 = np.concatenate([p2, p2[::-1]], axis=0)
    
    pf1 = np.fft.fft(pr1, axis=1)
    pf2 = np.fft.fft(pr2, axis=1)
    
    ap1 = np.abs(pf1)
    ap2 = np.abs(pf2)
    
    ap1_norm = (ap1 - ap1.min()) / (ap1.max() - ap1.min() + 1e-10)
    ap2_norm = (ap2 - ap2.min()) / (ap2.max() - ap2.min() + 1e-10)
    
    mse_val = mse(ap1_norm, ap2_norm)
    
    threshold = 0.1
    return 1 if mse_val < threshold else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Image processing program',
    )
    parser.add_argument('command', help='Command')
    parser.add_argument('parameters', nargs='*')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    if args.command in ['mse', 'psnr', 'ssim', 'compare']:
        img1 = skimage.io.imread(args.input_file)
        img2 = skimage.io.imread(args.output_file)
        
        if len(img1.shape) == 3:
            img1 = img1[:, :, 0]
        if len(img2.shape) == 3:
            img2 = img2[:, :, 0]
        
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        if args.command == 'mse':
            result = mse(img1, img2)
            print(f"{result:.6f}")
        elif args.command == 'psnr':
            result = psnr(img1, img2)
            print(f"{result:.6f}")
        elif args.command == 'ssim':
            result = ssim(img1, img2)
            print(f"{result:.6f}")
        elif args.command == 'compare':
            result = compare_images(img1, img2)
            print(result)
        exit(0)
    
    if args.command in ['median', 'gauss', 'bilateral']:
        img = skimage.io.imread(args.input_file)
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = img.astype(np.float64) / 255.0
        
        if args.command == 'median':
            rad = int(args.parameters[0])
            res = median_filter(img, rad)
        elif args.command == 'gauss':
            sigma_d = float(args.parameters[0])
            res = gaussian_filter(img, sigma_d)
        elif args.command == 'bilateral':
            sigma_d = float(args.parameters[0])
            sigma_r = float(args.parameters[1])
            res = bilateral(img, sigma_d, sigma_r)
        
        res = np.clip(res, 0, 1)
        res = np.round(res * 255).astype(np.uint8)
        skimage.io.imsave(args.output_file, res)
        exit(0)
    
    img = skimage.io.imread(args.input_file)
    img = img / 255
    if len(img.shape) == 3:
        img = img[:, :, 0]

    if args.command == 'mirror':
        res = mirror(img, args.parameters[0])

    elif args.command == 'extract':
        left_x, top_y, width, height = [int(x) for x in args.parameters]
        res = extract(img, left_x, top_y, width, height)

    elif args.command == 'rotate':
        direction = args.parameters[0]
        angle = int(args.parameters[1])
        res = rotate(img, direction, angle)

    elif args.command == 'autocontrast':
        res = autocontrast(img)

    elif args.command == 'fixinterlace':
        res = fixinterlace(img)

    res = np.clip(res, 0, 1)
    res = np.round(res * 255).astype(np.uint8)
    skimage.io.imsave(args.output_file, res)