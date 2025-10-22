# Internship Assesment Task: Ply Colorizer

## Goal

Write a small command line tool that will re-colorize a ply file given a bitmap file.
It should accept the name of the ply file and the bitmap file as parameters and output a new ply file that has the colors from the bitmap applied to the vertices in the ply file.

Bonus: there is also a LUT file included that can be used for color correction. It contains normalization factors for each color channel at different X and Z coordinates.

## Resources

- A ply file with its colors stripped: (f00014_20250910T151832_gray)
- A bmp file containing a colored 2D top-down view of the geometry found in the ply file: (f00014_20250910T151832.bmp)
- A colorized ply file to see how the result could look like: (f00014_20250910T151832.ply)
- A LUT file (ms_lut) that samples the function _C(X, Z) |-> (R, G, B, NIR, W5, W6, NIR2)_

## Limitations

- The x/y bounds of the ply file and the bmp are roughly equivalent. If they don't match up perfectly, it is ok to slightly overscan or underscan.
- The contents of the ply file and the bmp are not _exactly_ matching, so some misalignment is to be expected.
- You can pick any programming language you feel comfortable with, as long as you include instructions on how to compile and run the code.
- You are allowed to use LLM-generated code, but you should be prepared to understand and can reason about every line in a follow-up interview.

## Assessment 

Your solution will be judged according to the following properties:
- Applicability of the deliverable (how easy is it for us to get it working according to your instructions)
- Correctness of the functionality (does it achieve the goal)
- Code structure and clarity (how well can we read and understand your code; is it well structured and reflects the best practices of the language you chose)
- Whether the bonus task was also completed

Note that we value code clarity over 'cleverness' and over doing as much as possible in as few lines of code as possible.