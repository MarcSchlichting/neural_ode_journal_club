# Journal Club (Presentation) - Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations
This is repository contains the code that was used for the presentation in NEPR 280 on the paper *Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations*. The entire presentation was done using Grant Sanderson's (aka 3Blue1Brown) **Manim** library (not the community edition). Below are some quick hints on requirements and commands.

## Requirements
I made a couple of small tweaks to Grant's *Manim* by forking his repo. If you are using Grant's version, you will run into issues with the phase plots and using a remote for presenting won't work.

To install my fork of Manim, the following commands are necessary:
```bash
git clone https://github.com/MarcSchlichting/manimPresent.git
cd manimPresent
git checkout manim-present  # I work on a separate branch that is not the master branch
pip install -e .            # install the package in editable mode
```
You also need to have a local LaTex installation as nearly all text and math rendering relies on Latex. Plenty of documentation is available online for various platforms.

> Note: Pylance in VS code had difficulty with linting, so you have to manually add the `/.../manimPresent/manimlib/` directory to paths that Pylance uses for linting.


## Running the Presentation
The entire presentation is contained in `presentation.py`. Running the presentation in *presenter* mode gives a powerpoint-like feeling and the presentation is paused at defined points in the code (`self.wait()`). The next animation can be triggered via a right arrow click on the keyboard or the page down key (this is what Logitech presenters use).

```bash
manimgl presentation.py -p
```

> Note: The first time running the presentation might take a bit longer as all the LaTex renderings take some time.

## Rendering the Video
Rendering a single output video can be done using the following command:
```bash
manimgl presentation.py -w --uhd
```
> Note: Different resolutions can be selected. `--uhd` is 4k resolution, `--hd` is 1080p, etc. For full documentation see `manimgl --help`.
