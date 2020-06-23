# Computer vision

Computer vision project for personal learning.
___

## Documentation

The applications are located at `apps` directory and each one has it's own `readme.md` file.

### Face mask recognition and tracking
Detects multiple people and verifies if each person is wearing a face mask or not in real time.

___
## Getting started

### Requirements
- Python >=3.6 (CPython), PIP
- OS: Windows, Linux, Mac
- Minimum CPU dual core @1.9GHz

### Instalation
Install the dependencies and download the datasets/models.

#### Dependencies
```
pip install -r requirements/requirements.txt
```
**Note:** if you get an error when installing the `dlib` package, follow this [stackoverflow answer](https://stackoverflow.com/a/49538054).

#### Download datasets/models
1. Download the datasets/models from [pyimagesearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).
2. Change the directory and filenames by the `conf/settings.py` file.
___

## Running
### Starting the project
#### Typical
```
python main.py --settings=conf.settings
```
#### Using custom settings
You can override the `conf/settings.py` file by a custom one in `conf/env` (it will by ignored by git).
```
python main.py --settings=conf.env.dev
```
### Training the models
Usage of www.pyimagesearch.com scripts.
```
python train_mask_detector.py --dataset data/input/dataset
```
___

## Glossary

| Term            | Description |
| --------------- | ----------- |
| Computer vision | Science dicipline that make computers understand images of the real world. |
| SSD             | Single shot detector. |
| Bounding box    | Subarea in a image that may contain a detected object. |
| ROI             | Region of interest. See *Bounding box*. |
| CNN             | Convultional Neuronal Network. It's the preferred kind of network for computer vision. |
| Centroid        | Center point of a geometry object. |
| Mobilenet       | An effitient CNN for mobile vision applications. |
___

## Code Styling
Code styling is mandatory. The standards are the following:
- *Clean Code* by Robert C. Martin
- [Pocoo styleguide](https://flask.palletsprojects.com/en/1.1.x/styleguide/)
- PEP 20
- English

#### Exceptions:
- The maximum line length is **120 characters**, but you must try to decrease it.
- Use [type hints](https://docs.python.org/3/library/typing.html) whenever is possible.

Follow the `.pylintrc` file
___

## Credits
The models, dataset and algorithms are from www.pyimagesearch.com tutorials.
