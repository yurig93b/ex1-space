import os

from matcher import Matcher

IMG_DIR = 'imgs'

def main():
    for img1 in os.listdir("imgs"):
        for img2 in os.listdir("imgs"):
            if img1 == img2:
                continue
            m = Matcher(img1, img2)
            m.detect_circles()
            m.find_similarities()
            os.chdir('..')


if __name__ == "__main__":
    main()