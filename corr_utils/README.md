# CORR Utils

This (for now local) package contains utilities for working with the CORR. It makes use of the minimal package structure as defined [here](https://python-packaging.readthedocs.io/en/latest/minimal.html).

> If questions or requests arise, please contact [Noel Kronenberg](mailto:noel.kronenberg@charite.de).

## Structure

- ```extraction```: extracting data from a remote location
- ```covariate```: deriving variables
- ```analysis```: performing analyses
- ```ml```: building machine learning models

## Set-up

```bash
cd noel_thesis/corr_utils
pip install -e .
```

## Example Usage

```Python
# ...
import corr_utils.covariate as covariate
df_sap_lab_cleaned = covariate.extract_df_data(df_sap_lab, col_dict={'c_falnr':'case_id'})
```

## Testing

```bash
cd noel_thesis/corr_utils
python setup.py test
```

## Documentation

The documentation can be found here: [noelkronenberg.github.io/corr_utils/](noelkronenberg.github.io/corr_utils/). It can be updated with [Sphinx](https://www.sphinx-doc.org/en/master/index.html):

```bash
cd noel_thesis/corr_utils/docs
make clean
make html
```

## Git

Initialize:

```bash
git init # initialize folder as a Git repository
git remote add origin https://github.com/noelkronenberg/corr_utils.git # link local repository to the GitHub repository
git branch -M main
```

Update:

```bash
git pull # check for updates
git add . # add files to the staging area
git commit -m "Commit message" # commit changes
git push -u origin main # push  changes to master branch of the GitHub repository
```
