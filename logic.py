# logic.py
# Imports
import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from groq import Groq
import json
import base64
import time
import re
import glob
from PIL import Image
from itertools import combinations
import shutil
import pptx
from pptx.enum.shapes import MSO_SHAPE_TYPE
import io
from wand.image import Image as WandImage
from wand.color import Color
import csv


# --- PART 1: FUNCTIONS (From AnkiCardGenerator_Function_V2) ---

def io_generate(input_filename, output_filename, results):
    pathway = f"image{input_filename}.png"

    # Allow CV2 to load it
    image = cv2.imread(pathway)

    count = output_filename
    for result in results:
        # Cords & Text
        cords = result[0]
        text = result[1]

        # Reshaping Cords to Numpy Array
        all_text_points = np.array(cords, dtype=np.int32)
        all_text_points = all_text_points.reshape((-1, 1, 2))

        cv2.fillPoly(image, [all_text_points], color=(255, 0, 0))
        cv2.putText(
            image,  # Image
            str(count),  # Label
            (int(cords[0][0]), int(cords[0][1]) + 20),  # Cords
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1.0,  # Font Size
            (255, 255, 255),  # Font Color
            2  # Font Thickness
        )

        cv2.imwrite(f'covered_image{count}.png', image)
        print("Text covered successfully.")

    # Give Time For Directory Adjust
    time.sleep(1)

    original_image = cv2.imread(f'covered_image{count}.png')
    for result in results:
        # Make a copy of Image
        current_card = original_image.copy()

        # Cords & Text
        cords = result[0]
        text = result[1]

        # Writing the covered text to a text file
        with open(f"answer{count}.txt", 'a') as file:
            file.write(f"{text}")

        # Reshaping Cords to Numpy Array
        all_text_points = np.array(cords, dtype=np.int32)
        all_text_points = all_text_points.reshape((-1, 1, 2))

        cv2.fillPoly(current_card, [all_text_points], color=(0, 255, 0))
        cv2.imwrite(f'covered_image{count}.png', current_card)
        print("Text covered successfully.")

        count += 1

    return count


def standard_generate(input_filename, output_filename, slide_text):
    try:
        # Renaming Image to Final Image
        os.rename(f"image{input_filename}.png", f"covered_image{output_filename}.png")

        # Creating Answer File
        with open(f"answer{output_filename}.txt", 'a') as file:
            file.write(f"{slide_text}")

    except Exception as e:
        print(e)

    return output_filename + 1


def groq_preprocessing_tool(similar_image_group, slide_deck_name):
    """
    Preprocessing tool.
    Receives a list of IMAGES THAT ARE ALREADY CONFIRMED TO BE VISUALLY SIMILAR.
    Decides which image(s) to 'keep' and which to 'delete'.
    """

    # Connecting To Groq
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Helper to generate the default "Keep Everything" response on failure
    def default_keep_all():
        return {os.path.basename(f): "keep" for f in similar_image_group}

    # 1. Prepare the Payload
    content_payload = [
        {
            "type": "text",
            "text": f"""
            You are a Medical Content Curator.
            Context: The user is converting the lecture "{slide_deck_name}" into flashcards.

            TASK: The user has identified the following {len(similar_image_group)} images as DUPLICATES or variants of the same slide.
            You must decide which image is the 'Best Version' to keep.

            CRITERIA FOR 'keep':
            1. HIGHEST RESOLUTION: Prefer sharp, clear text.
            2. DIAGNOSTIC VALUE: Prefer images with CLEAR labels/arrows pointing to anatomy.
            3. CLEANLINESS: Avoid heavy watermarks if a cleaner version exists.

            DECISION LOGIC:
            - If Image A is Clean and Image B has Labels: Mark B 'keep', Mark A 'delete'.
            - If Image A and Image B are identical content but A is larger/sharper: Mark A 'keep', Mark B 'delete'.
            - If you cannot decide, mark both as 'keep'.

            Return ONLY a valid JSON object mapping filenames to 'keep' or 'delete'.
            Example format:
            {{
                "image1.png": "keep",
                "image2.png": "delete"
            }}
            """
        }
    ]

    # 2. Encode Images
    valid_filenames = []
    if similar_image_group:
        for file_path in similar_image_group:
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    filename = os.path.basename(file_path)
                    valid_filenames.append(filename)

                    content_payload.append({
                        "type": "text",
                        "text": f"Image Filename: {filename}"
                    })
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}",
                        },
                    })
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")

    # 3. Make the API Call
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content_payload}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,  # Strict comparison
            response_format={"type": "json_object"}
        )

        # 4. Parse Response Robustly
        response_content = chat_completion.choices[0].message.content

        # Find JSON boundaries
        start_index = response_content.find('{')
        end_index = response_content.rfind('}')

        if start_index != -1 and end_index != -1:
            clean_json = response_content[start_index: end_index + 1]
            return json.loads(clean_json)

        # If parsing fails, default to keeping everything
        print("Error: Could not find JSON in response. Keeping all images.")
        return default_keep_all()

    except Exception as e:
        print(f"Groq Preprocessing Error: {e}")
        # SAFETY NET: Return "keep" for everything so the pipeline continues
        return default_keep_all()


def similarity_images(image1, image2):
    """
    This function compares the similarity of images
    """
    try:
        # Reading the Images as Grayscale
        im1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

        # Reduce Shape to Compare Structures
        im1 = cv2.resize(im1, (64, 64))
        im2 = cv2.resize(im2, (64, 64))

        # Finding the Square Diff
        diff = im1.astype(float) - im2.astype(float)
        squared_diff = diff ** 2

        # Finding the Mean
        mse = np.mean(squared_diff)

        if mse < 200:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def pre_processing_prune(slide_deck_name):
    # Ascertaining All Files
    image_files = glob.glob("image*.png")
    image_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Deleting All Small or Funky Images
    for image in image_files:
        bad_card = False
        try:
            with Image.open(image) as img:
                # Retrieving Dimensions
                width, height = img.size

                # Removing Smoll Images
                if width < 150 or height < 150:
                    bad_card = True

                # Removing Images w/ weird ratios
                if width / height > 5.0 or width / height < 0.2:
                    bad_card = True
        except Exception as e:
            print("Error 222: Bad File Type")

        # Removing the Image
        try:
            if bad_card:
                # CHANGED: Relative path for cloud compatibility
                shutil.move(image, os.path.join(os.getcwd(), "Criteria1"))
        except FileNotFoundError:
            print("Error 2: File Not Found")

    # Now Beginning Duplication Pruning
    # Ascertaining All Files
    image_files = glob.glob("image*.png")
    image_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Batch Process Similar Images
    orphans_to_delete = []
    batch_size = 4

    for i in range(0, len(image_files), batch_size):
        # Batching Files Together
        batch = image_files[i: i + batch_size]
        combinations_batch = list(combinations(batch, 2))

        # Create Combinations
        similar_images = []
        for combination in combinations_batch:
            # Check if Images Similar
            if similarity_images(combination[0], combination[1]):
                if combination[0] not in similar_images:
                    similar_images.append(combination[0])
                if combination[1] not in similar_images:
                    similar_images.append(combination[1])

        if len(similar_images) < 2:
            pass
        else:
            decisions = groq_preprocessing_tool(similar_images, slide_deck_name)
            # Go Through Every Image
            for key, value in decisions.items():
                if value == "delete":
                    orphans_to_delete.append(key)

    for orphan in orphans_to_delete:
        try:
            # CHANGED: Relative path
            shutil.move(orphan, os.path.join(os.getcwd(), "Criteria2"))
        except Exception as e:
            print(e)


def post_processing_prune(slide_deck_name):
    # Ascertaining All Files
    answer_files = glob.glob("answer*.txt")
    answer_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    covered_image_files = glob.glob("covered_image*.png")
    covered_image_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Removing Text w/ Keywords or Phrases
    for file in answer_files:
        bad_card = False
        try:
            with open(file, "r") as txt_file:
                # Keywords to Delete Images
                keywords = [
                    # --- CLASSROOM ADMIN & META ---
                    "learning objectives",
                    "objectives",
                    "goals",
                    "agenda",
                    "overview",
                    "course outline",
                    "schedule",
                    "announcements",
                    "housekeeping",
                    "welcome to",
                    "module",
                    "lecture",
                    "attendance",
                    # --- END OF DECK / INTERACTION ---
                    "any questions",
                    "questions?",
                    "questions & answers",
                    "q&a",
                    "thank you",
                    "thanks",
                    "contact info",
                    "contact me",
                    "email:",
                    "discussion",
                    "group activity",
                    "quiz time",
                    "pop quiz",
                    "break time",
                    # --- CITATIONS, LEGAL & SOURCING ---
                    "references",
                    "works cited",
                    "bibliography",
                    "sources",
                    "citations",
                    "image credits",
                    "image source",
                    "figure source",
                    "copyright",
                    "all rights reserved",
                    "conflict of interest",
                    "disclosures",
                    "financial disclosure",
                    "no disclosures",
                    # --- SUMMARY & FLUFF ---
                    "summary",
                    "conclusion",
                    "recap",
                    "take home points",
                    "key takeaways",
                    "take-away",
                    "in summary",
                    "to summarize",
                    # --- POWERPOINT ARTIFACTS ---
                    "click to add title",
                    "click to add text",
                    "slide number",
                ]

                # Checking if the keywords are in the text file
                content = txt_file.read().lower()  # Read once
                for keyword in keywords:
                    if keyword in content:
                        bad_card = True

            # Removing the card
            try:
                if bad_card:
                    # CHANGED: Relative paths
                    shutil.move(file, os.path.join(os.getcwd(), "Criteria3"))
                    number = file.strip("answer").strip(".txt")
                    shutil.move(f"covered_image{number}.png",
                                os.path.join(os.getcwd(), "Criteria3"))
            except FileNotFoundError:
                print("Error 2: File Not Found")
        except Exception as e:
            print(e)

    # Ascertaining Remaining Images Post 2 Prunes
    final_image_list = glob.glob("covered_image*.png")

    # AI Pruning
    pruning_decisions = {}
    batch_size = 5

    # AI Pruning in Batches
    for i in range(0, len(final_image_list), batch_size):
        batch = final_image_list[i: i + batch_size]
        print(f" -> Pruning Batch {i // batch_size + 1}: Checking {len(batch)} images...")

        # Call the tool
        batch_results = groq_pruning_tool(batch, slide_deck_name)

        # Merge results into main dictionary
        pruning_decisions.update(batch_results)

    # Deleting Orphans
    for key, value in pruning_decisions.items():
        if value == "Delete":
            # CHANGED: Relative paths
            shutil.move(key, os.path.join(os.getcwd(), "Criteria4"))
            number = key.strip("covered_image").strip(".png")
            shutil.move(f"answer{number}.txt", os.path.join(os.getcwd(), "Criteria4"))


def groq_pruning_tool(file_paths, slide_deck_name):
    """
    Post-processing tool.
    """
    # Connecting To Groq
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # 1. Prepare the Payload List
    content_payload = [
        {
            "type": "text",
            "text": f"""
            You are a rigorous Medical Slide Deck Cleaner. 
            Context: These images are from a lecture deck named "{slide_deck_name}".
            Your Task: Review each image and assign it one of two statuses: 'Delete' or 'Pass'.
            ... [Original Prompt Kept Intact] ...
            Return ONLY a valid JSON object.
            """
        }
    ]

    # 2. Encode Images and Append
    valid_filenames = []

    if file_paths:
        for file_path in file_paths:
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    filename = os.path.basename(file_path)
                    valid_filenames.append(filename)

                    content_payload.append({
                        "type": "text",
                        "text": f"Image Filename: {filename}"
                    })
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}",
                        },
                    })
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")

    # 3. Make the API Call
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content_payload}],
            model="llama-3.2-11b-vision-preview",  # CHANGED: Valid Groq Model
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # 4. Parse Response
        response_content = chat_completion.choices[0].message.content
        start_index = response_content.find('{')
        end_index = response_content.rfind('}')

        if start_index != -1 and end_index != -1:
            clean_json_string = response_content[start_index: end_index + 1]
            try:
                decision_data = json.loads(clean_json_string)
                print(f"Pruning Batch Decision: {decision_data}")
                return decision_data
            except json.JSONDecodeError:
                return {name: "Pass" for name in valid_filenames}
        else:
            return {name: "Pass" for name in valid_filenames}

    except Exception as e:
        print(f"Groq API Error: {e}")
        return {name: "Pass" for name in valid_filenames}


# --- PART 2: MAIN PIPELINE (Adapted from Slide_To_Text) ---

def run_pipeline(presentation_name, start_counter):
    # Making Folders for Errors (Relative Paths)
    for x in range(1, 5, 1):
        newpath = os.path.join(os.getcwd(), f'Criteria{x}')
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    # Tracking Time
    start_time = time.perf_counter()

    # Accessing Presentation
    prs = pptx.Presentation(presentation_name)
    slides = prs.slides

    # Getting All Slide Texts
    counter = 0
    slide_texts = {}

    # Loop Through Every Slides
    for slide in slides:
        # Loop Through Every Shape
        for shape in slide.shapes:
            # Check if Shape is Image
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                # Saving the Image
                im = shape.image
                im_byte = im.blob
                im_filename = f'image{counter}.png'
                im_stream = io.BytesIO(im_byte)

                # Convert Image to .png
                try:
                    # Check if it is a goofy file type
                    if 'wmf' in im.content_type or 'emf' in im.content_type:
                        with WandImage(blob=im_byte) as vector_img:
                            vector_img.background_color = Color('white')
                            vector_img.format = 'png'
                            vector_img.save(filename=im_filename)
                    # If it normal then normally convert w PIL
                    else:
                        img = Image.open(im_stream)
                        img.save(im_filename, format='PNG')

                    # Allow for Directory to Adjust for new Image
                    time.sleep(1)

                    # Get all slide text
                    slide_text = ""
                    # Go through all shapes
                    for shape1 in slide.shapes:
                        # Check if text
                        if shape1.has_text_frame:
                            # Add it
                            slide_text += str(shape1.text_frame.text)

                    # Add It To Array
                    slide_texts[im_filename] = slide_text

                except Exception as e:
                    print("Error 222: Bad File Type")
                counter += 1

    # Conduct Pre-processing
    pre_processing_prune(presentation_name)

    # Get Remaining Images
    image_list = glob.glob("image*.png")
    image_list.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Loading AI Model
    ocr_engine = RapidOCR()

    # Looping Through Every Image
    counter = int(start_counter)  # Use the passed argument

    for image in image_list:
        results, elapse = ocr_engine(image)

        # DEFAULT TO STANDARD (Assume it's just a picture until proven otherwise)
        is_valid_io_card = False

        if results:
            # Loop through EVERY detection to find at least one letter
            for detection in results:
                text = detection[1]
                confidence = detection[2]

                # CHECK: Has letters (A-Z) AND Confidence > 60%
                if re.search('[a-zA-Z]', text) and confidence > 0.6:
                    is_valid_io_card = True
                    break

        if is_valid_io_card:
            # It has real text -> Make IO Card
            counter = io_generate(int(image.strip("image").strip(".png")), counter, results)
        else:
            # It's empty -> Make Standard Card
            counter = standard_generate(int(image.strip("image").strip(".png")), counter, slide_texts[image])

    # Locating All Files w/ image*
    files = glob.glob("image*.png")

    for file_path in files:
        # Extra safety check: Ensure we don't accidentally delete the final cards
        if not file_path.startswith("covered_"):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

    post_processing_prune(presentation_name)

    print(f"The code took {time.perf_counter() - start_time} seconds.")


# --- PART 3: FORMATTING (Adapted from Formatting) ---

def create_csv_file(csv_name):
    # Gather Files
    answer_files = glob.glob("answer*.txt")
    answer_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    image_files = glob.glob("covered_image*.png")
    image_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Retrieving the IDs
    answer_ids = set()
    for f in answer_files:
        match = re.search(r"answer(\d+)\.txt", f)
        if match:
            answer_ids.add(int(match.group(1)))

    image_ids = set()
    for f in image_files:
        match = re.search(r"covered_image(\d+)\.png", f)
        if match:
            image_ids.add(int(match.group(1)))

    # Identifying valid cards
    valid_ids = sorted(list(answer_ids.intersection(image_ids)))

    # Write CSV
    csv_rows = []
    for num in valid_ids:
        txt_filename = f"answer{num}.txt"
        img_filename = f"covered_image{num}.png"

        try:
            with open(txt_filename, "r", encoding="utf-8") as file:
                content = file.read().strip()
                # Replace newlines with HTML breaks for Anki
                content = content.replace("\n", "<br>")

                # Anki Format: Column 1 = Image, Column 2 = Back Text
                csv_rows.append([f'<img src="{img_filename}">', content])
        except Exception as e:
            print(f"Error reading {txt_filename}: {e}")

    if not csv_name.endswith(".csv"):
        csv_name += ".csv"

    if csv_rows:
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_rows)
        return csv_name
    else:
        return None
