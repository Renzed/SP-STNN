from PIL import Image
import os

if __name__ == "__main__":
    path = input('Enter path from here: ')
    with Image.open(path) as im:
        os.mkdir(path[:-4])
        for i in range(im.n_frames):
            im.seek(i)
            if len(str(im.n_frames)) == 1:
                im.save(f"{path[:-4]}/frame_{i}.gif", 'GIF')
            if len(str(im.n_frames)) == 2:
                im.save(f"{path[:-4]}/frame_{i:02}.gif", 'GIF')
            if len(str(im.n_frames)) == 3:
                im.save(f"{path[:-4]}/frame_{i:03}.gif", 'GIF')
            if len(str(im.n_frames)) == 4:
                im.save(f"{path[:-4]}/frame_{i:04}.gif", 'GIF')
