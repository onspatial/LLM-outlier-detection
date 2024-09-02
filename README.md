
# Dataset to be used:
- Train Dataset on [OSF](https://osf.io/gbhm8/) and description on [this paper](https://dl.acm.org/doi/abs/10.1145/3589132.3625592)
- Test Dataset on [OSF](https://osf.io/rxnz7/)


# Steps to reproduce the results:

0. Download the Checkin.tsv to data/PofL
1. To generate the prompt for pattern-of-life data
    - `python3 1-generate-prompt-pattern-of-life.py`
2. To generate the prompt for geolife data
    - `python3 2-generate-prompt-geolife.py`
3. Run openai API based on generated prompt
    - `python3 3-openai.py --data geolife --model gpt-3.5-turbo-16k-0613 --with-hint`

# Sample Prompt: 

![image](https://github.com/user-attachments/assets/fb0cec3a-9b1d-4c5d-b8f3-ee2a513eb738)

![image](https://github.com/user-attachments/assets/7e0ea807-caae-44dd-9a56-5b41cc9fd34b)
