import os
import matplotlib.pyplot as plt
from PIL import Image

class FigureSaver:
    def __init__(self, base_directory=".", default_dpi=500, default_format="png"):
        """
        Initializes the FigureSaver object.
        """
        self.base_dir = base_directory
        self.default_dpi = default_dpi
        self.default_format = default_format

    def save(self, folder_name, file_name, fig=None, dpi=None, fmt=None, **kwargs):
        """
        Saves a matplotlib figure, creating the target directory if it doesn't exist.
        """
        save_dpi = dpi if dpi is not None else self.default_dpi
        save_fmt = fmt if fmt is not None else self.default_format
        
        full_dir_path = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)

        clean_file_name = f"{file_name}.{save_fmt}"
        full_file_path = os.path.join(full_dir_path, clean_file_name)

        if fig is None:
            fig = plt.gcf()

        bbox = kwargs.pop('bbox_inches', 'tight')
        facecolor = kwargs.pop('facecolor', 'white')

        fig.savefig(full_file_path, dpi=save_dpi, format=save_fmt, bbox_inches=bbox, facecolor=facecolor, **kwargs)
        print(f"✅ Figure saved to: {full_file_path}")

    def create_gif(self, input_folder, gif_name, output_folder=None, duration=100, loop=0):
        """
        Compiles a directory of PNG images into a GIF.
        """
        full_input_dir = os.path.join(self.base_dir, input_folder)
        
        if not os.path.exists(full_input_dir):
            raise FileNotFoundError(f"❌ Input folder not found: {full_input_dir}")

        target_folder = output_folder if output_folder else input_folder
        full_output_dir = os.path.join(self.base_dir, target_folder)
        
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        images = [img for img in os.listdir(full_input_dir) if img.endswith('.png')]
        images.sort()

        if not images:
            print(f"⚠️ No PNG images found in {full_input_dir} to make a GIF.")
            return

        frames = []
        for image in images:
            img_path = os.path.join(full_input_dir, image)
            frames.append(Image.open(img_path))

        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
        full_gif_path = os.path.join(full_output_dir, gif_name)

        frames[0].save(
            full_gif_path,
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=loop
        )
        print(f"🎬 GIF successfully created at: {full_gif_path}")

    def combine_gifs(self, gif1_name, gif2_name, output_name, subfolder=None):
        """
        Combines two GIFs side-by-side into a single GIF.
        The smaller frame is resized to match the height of the larger frame
        while maintaining its aspect ratio.
        """
        # Determine the directory to look in
        target_dir = os.path.join(self.base_dir, subfolder) if subfolder else self.base_dir
        
        # Helper to ensure .gif extension
        def format_gif_name(name):
            return name if name.endswith('.gif') else f"{name}.gif"

        gif1_path = os.path.join(target_dir, format_gif_name(gif1_name))
        gif2_path = os.path.join(target_dir, format_gif_name(gif2_name))
        output_path = os.path.join(target_dir, format_gif_name(output_name))

        try:
            gif1 = Image.open(gif1_path)
            gif2 = Image.open(gif2_path)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find a GIF file. {e.filename}")
            return

        if gif1.n_frames != gif2.n_frames:
            print("⚠️ Warning: GIFs have different number of frames. The shorter length will be used.")

        num_frames = min(gif1.n_frames, gif2.n_frames)
        new_frames = []
        
        print(f"🔄 Resizing and combining {num_frames} frames from each GIF...")

        for i in range(num_frames):
            gif1.seek(i)
            gif2.seek(i)
            
            frame1 = gif1.copy().convert("RGBA")
            frame2 = gif2.copy().convert("RGBA")
            
            width1, height1 = frame1.size
            width2, height2 = frame2.size
            
            target_height = max(height1, height2)
            
            if height1 < target_height:
                aspect_ratio = width1 / height1
                new_width1 = int(target_height * aspect_ratio)
                frame1 = frame1.resize((new_width1, target_height), Image.Resampling.LANCZOS)
            
            if height2 < target_height:
                aspect_ratio = width2 / height2
                new_width2 = int(target_height * aspect_ratio)
                frame2 = frame2.resize((new_width2, target_height), Image.Resampling.LANCZOS)
            
            final_width1, _ = frame1.size
            final_width2, _ = frame2.size
            
            combined_width = final_width1 + final_width2
            new_frame = Image.new('RGBA', (combined_width, target_height))
            
            new_frame.paste(frame1, (0, 0))
            new_frame.paste(frame2, (final_width1, 0))
            
            new_frames.append(new_frame)
        
        if new_frames:
            duration = gif1.info.get('duration', 100) 
            
            new_frames[0].save(
                output_path,
                save_all=True,
                append_images=new_frames[1:],
                duration=duration,
                loop=0, 
                optimize=True
            )
            print(f"✅ Successfully created combined GIF: {output_path}")
        else:
            print("❌ No frames were generated. Could not create GIF.")
            
    def save_memory_gif(self, frames, gif_name, output_folder=None, duration=100, loop=0):
        """
        Saves a list of in-memory PIL Image frames directly into a GIF.
        """
        target_folder = output_folder if output_folder else "."
        full_output_dir = os.path.join(self.base_dir, target_folder)
        
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
            
        full_gif_path = os.path.join(full_output_dir, gif_name)

        if frames:
            frames[0].save(
                full_gif_path,
                append_images=frames[1:],
                save_all=True,
                duration=duration,
                loop=loop
            )
            print(f"🎬 Memory GIF successfully created at: {full_gif_path}")
        else:
            print("❌ Error: Frame list is empty.")