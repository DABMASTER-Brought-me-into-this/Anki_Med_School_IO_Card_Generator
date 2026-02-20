# MedSchool IO Card Anki Generator 

This project was written in dedication of a special Medical School student. Medical School is very difficult and filled with lots of memorization. 

Many medical students use an application called Anki to help with this. One of the many things students have to be able to memorize are names of specific images and parts within those images. This application automates this process by reading a medical school's `.pptx` file and automatically generating flashcards on [slidemd.streamlit.app](https://slidemd.streamlit.app/).

The app does one of two things:
* **Image Occlusion (IO) Cards:** If the image has detectable text on it, the app will cover the text with blue and green squares. The green square represents what you are trying to answer, and the blue boxes cover the surrounding text to prevent cheating.
* **Standard Cards:** If the image does not have detectable text, the app will take text identified from the slide and put it on the back of the card.

### The 4 Pruning Criteria
Medical school slides have many images: some duplicates, some cartoons, some stock photos. To save the user time purging these files manually, 4 filtering criteria were created to output only relevant cards.

* **Criteria 1 (Dimensions):** The image must have a height and width of at least 150px, and a height:width ratio between 0.2 and 5.
  * *Ideal Case:* Designed to automatically remove dividers, UI arrows, and small formatting shapes.
* **Criteria 2 (Similarity Check):** Removes images with high structural similarities. Groq's Vision AI chooses which image to keep.
  * *Ideal Case:* Many slide decks have quizzes (one picture with the answer, one without). To ensure no duplication, Groq selects the best version. 
* **Criteria 3 (Keyword Exclusion):** Removes any IO or Standard cards that contain specific non-academic keywords.
  * *Ideal Case:* Some images are just decorative text to announce "Break Time" or "Agenda". AI was used to generate a list of these keywords, and if picked up, the card is purged.
* **Criteria 4 (AI Overview):** Groq reviews all remaining images in batches of 5.
  * *Ideal Case:* Groq will delete anything it deems irrelevant, duplicate (if criteria 2 failed), unclear, or purely formatting-based.

---

## Technical Debt & Upcoming Changes
This project currently works, but there are inefficiencies. The output quality and generation speed are actively being optimized.

**More specifics for nerds:**
* As of right now, the IO cards have the boxes drawn on one at a time directly to the disk. I will implement a future change to process this entirely in-memory (RAM) using Numpy arrays until the final export.
* Currently, the only compatible file type is PPTX. While other files can be converted to PPTX, I plan to support other common formats natively (PDF, GSlides, Keynote).
* To increase the readability and maintainability of the codebase, I will un-consolidate my files into a modular architecture.
* Currently, I am sending one API Request at a time. To increase speed, I will utilize Async runs. However, in consideration of free-to-use API limits, I will implement rate-limiting safeguards.
