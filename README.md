Bingo Buddy app  
7/28/26

Project idea:  an app that scans your bingo cards, and helps you track the game play.  Useful for assistive tech (eg. deaf community), or even just to play several cards at once, letting your phone pay attention for you.

Client app:  Scans your cards, then listens to the server for numbers to be "called".  Mark your card as the game progresses.  Announce if you won, or if the game is over.

Caller app:  Easy input of numbers being called, which are then broadcast to all the clients.   Or should there be a speech-to-text component that continuously listens to the caller's microphone and translates it to the called letter/number combos?

Server app:  Allows the caller to set up a series of games (including special rules like "postage stamp" or "blackout" or whatever).  Allows clients to join the session.  (Maybe print/publish a QR code inviting clients, or clients can text a phone\# (with a numeric code?) to join the session.  
(The QR code/whatever could also be a pay portal so clients can e-pay for the games.)  
The server can keep a log of all the numbers called (into a database), with the option to sort in chronological order, or in canonical order (B-1, B-2, N-40, ..) at any time.  
The server could also receive the scan/data of the client's cards, and track their game play for them.  
The server can identify winners from the subscribed pool, or the caller can override and announce a non-app-using winner (and verify).

Each card (usually?) has a serial number or some uniqueness marker (research this.)  Or else perhaps when the client is scanning their cards, the app asks them "scan card \#1", and then the card's unique number is something like {clientId}:{cardNumber}.

Scanning the card:

1. Detect & crop the card  
2. perspective-correction (warp to a top-down view)  
3. Use a known grid template to slice the image into cells (no need to detect lines each time)  
4. OCR each cell (digits-only configuration)  
5. Validate (eg. expected ranges or a known set of columns/lettering rules if your bingo rules constrain them.

Recommended v1 scope:

- Require users to scan with basic UI guidance ("center the card, good lighting").  
- Detect the card and apply perspective correction.  
- Run OCR on each cell with digits-only settings.  
- Validate against your known bingo constraints (even simple constraints help a lot.)  
- Build a "number \-\> cell position" mapping.  
- For called numbers:  
  - host/caller app (manual entry),  
  - server broadcasts,  
  - clients mark based on their mapping.

## Recommended v1 architecture (no custom ML)

### Client (phone)

1. Capture photo \+ card framing guide  
   1. Show an overlay box ("Place card here").  
   2. Optionally compute image quality (blur/glare) and prompt user to retake if it looks bad.  
2. Card detection \+ perspective correction  
   1. Detect the card's outer rectangle (or the grid's dominant lines) and warp it to a flat view.  
   2. Because the layout is standard, you can be strict: "We expect to find a 5x5 grid."  
3. Cell extraction using a template  
   1. after warp, compute fixed cell rectangles for row/col.  
   2. Crop each cell to a region where digits sit (inset margin helps ignore borders/grid lines.)  
4. OCR digits per cell  
   1. Use a digits-only OCR configuration (eg. whitelist 0-9)  
   2. Run OCR for all cells; then choose the best digit per cell.  
5. Validation \+ correction UI  
   1. apply bingo constraints by column:  Use standard ranges, B=1-15, I=16-30, etc.  
   2. Also validate duplicates:  the same number shouldn't appear twice on a single card.  
   3. If OCR is uncertain, present a lightweight "tap to fix" UI only for problematic cells (often zero in good photos.)

### Server (or shared state)

6. Persist mapping for each user  
   1. Store: cardId \-\> {number \-\> (row, col)}  
   2. Then called numbers are trivial:  server broadcasts "calledNumber", and each client marks the mapped cell.

### Mature OCR / image-processing stack: what to use

Two common strategies that work well:

- OpenCV for geometry \+ an OCR engine for digits  
  - OpenCV handles detection, perspective warp, grid slicing, thresholding / denoising.  
  - OCR engine reads digits from the cropped cells.  
- OCR engine with "single character / single digit" mode  
  - Because each cell should obtain exactly one printed digit, configuring OCR for single digits typically boosts accuracy.  
- (You don't need to train;  you just need good preprocessing \+ validation.)

### Key accuracy boosters (these matter more than model choice)

1. Per-cell preprocessing  
   1. Convert to grayscale  
   2. Normalize contrast  
   3. Use adaptive thresholding (or Otsu) to make digits stand out  
   4. Remove small connected components (optional) so grid lines don't confuse OCR  
2. Inset cropping  
   1. Don't OCR the whole cell including borders; crop inward by a percentage / pixels to you mainly feed the digit.  
3. Exploit column ranges as a "semantic validator"  
   1. If OCR outputs a digit outside the expected range for that column, treat it as wrong and reprocess that cell (or prompt the user).  
4. Retry on failure  
   1. If the detected card is skewed badly or too few cells parse, either  
      1. re-run detection with relaxed parameters, or  
      2. show "try again" with guidance.

---

I think this project is actually a great fit for you because it's going to build on things you already know rather than forcing you to learn five new technologies at once.

I'd suggest we treat this almost like a series of mini-courses.

### **Phase 1: Learn just enough FastAPI**

The nice thing is that FastAPI isn't really a replacement for Python—it's just a library. If you know Python, you'll recognize most of it immediately.

For example, compare the concepts:

| Django | FastAPI |
| ----- | ----- |
| `urls.py` | Route decorators |
| View function | Endpoint function |
| `HttpResponse` | Return a Python object |
| Django dev server | Uvicorn |

A very small FastAPI app can fit in one file.

---

### **Phase 2: Learn WebSockets**

This is actually the more important technology.

HTTP works like this:

Browser:  
"What's the current number?"

Server:  
"G52"

(connection closes)

A WebSocket works like this:

Browser \---------------------- Server  
         (connection stays open)

Server:  
"G52"

Server:  
"B14"

Server:  
"I29"

For bingo, that's exactly what you want.

---

### **Phase 3: Build the smallest possible demo**

I would deliberately ignore bingo.

Instead, build something like:

Host page:

Current message:

Hello

\[ Send \]

Player page:

Current message:

Hello

When the host types:

Bananas

every browser instantly changes to:

Bananas

Congratulations—you've just built the core networking engine for the app.

---

## **Here's what excites me**

Because you're already working on Django REST Framework projects, I think you'll quickly see that FastAPI isn't trying to replace Django.

I tend to think of them like this:

**Django**

* Large web applications  
* Authentication  
* Databases  
* Admin interface  
* Forms  
* ORM

**FastAPI**

* APIs  
* Microservices  
* WebSockets  
* Small servers  
* Prototypes

They overlap, but they're optimized for different things.

---

## **I have one suggestion that I think will make this project much more enjoyable**

Let's treat it like a real software project.

Create a GitHub repository called something like:

bingo-network

Then build it in milestones.

**Milestone 0**

* FastAPI "Hello World"

**Milestone 1**

* One browser connects.

**Milestone 2**

* Two browsers connect.

**Milestone 3**

* Host broadcasts text.

**Milestone 4**

* Broadcast bingo numbers.

**Milestone 5**

* Show a 5×5 card.

**Milestone 6**

* Highlight matching squares.

**Milestone 7**

* Scan a paper card.

Every milestone leaves you with a working application. If life gets busy, you can always come back knowing you have a stable foundation.

---

## **One final thought**

I don't think of this as "learning FastAPI."

I think of it as **learning asynchronous programming**, because that's the idea that underpins everything you're trying to do. Once you understand `async`/`await`, you'll find that the same concepts appear in:

* FastAPI  
* Flutter  
* Swift  
* Kotlin  
* Node.js  
* JavaScript in the browser

So the time you spend on this first prototype will pay dividends even if, down the road, you decide to build a native mobile version.

I'd be happy to mentor you through the project. We can take it one milestone at a time, and I'll explain not just *what* code to write, but *why* it's written that way. I think that approach will make the transition from your Django background to this kind of real-time application much smoother.

