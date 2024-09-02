# LLM-outlier-detection
############

0. Download the Checkin.tsv to data/PofL

1. To generate the prompt for pattern-of-life data

python3 1-generate-prompt-pattern-of-life.py

2. To generate the prompt for geolife data

python3 2-generate-prompt-geolife.py

3. Run openai API based on generated prompt

python3 3-openai.py --data geolife --model gpt-3.5-turbo-16k-0613 --with-hint


![image](https://github.com/user-attachments/assets/fb0cec3a-9b1d-4c5d-b8f3-ee2a513eb738)

![image](https://github.com/user-attachments/assets/24800777-f7e7-4e76-9514-10aba90ca72c)
