
project_name = "edm25"

# the standard output folder. Doesn't need to be changed usually
outputpath = "../output/"
outputpath = "./output/"

# Define the database dump you want to analyse. Usually it is the most recent dump
dump = 'dump20240826' # all data
dump = 'dump20241104'
dump = 'dump20241111'
dump = 'dump20241118'
dump = 'dump20241125'
dump = 'dump20241202'
dump = 'dump20241209'

dump = 'dump20240826' # all data


# Choose a semester or assign the value 'all' to obtain data from all semesters that are in the database dump
#semester = 'SS2021'
#semester = 'SS2021' ### CSCW'25 study
#semester = 'SS2021'
semester = 'WS2022_23' ### CSCW'25 study
#semester = 'WS2023_24' ### 
#semester = 'SS2024'
###semester = 'WS2024_25' ### 
#semester = 'SS2025'
# semester = 'all'

semesters = [
    'SS2021',
    'WS2021_22',
    'SS2022',
    'WS2022_23',
    'SS2023',
    'WS2023_24',
    'SS2024',
    'WS2024_25',
]

hashit = False

# Define teacher IDs
teacher_ids = [
    [10863, 6280, 37, 21, 7, 9, 8843, 6, 5, 8800, 8847, 8848, 8846],
    [10863, 6280, 37, 21, 7, 9, 8834, 6, 5, 8959],
    [10863, 6280, 37, 21, 7, 9, 8834, 6, 5]
]

# SETTINGS
# Change this time periods if you are processing data of another semester

# Define periods for T1 (Kennenlernphase)
period_1_arr = {
    'SS2021': ["21-16", "21-17"],             # 27.04.21 - 08.05.21, KW 
    'WS2021_22': ["21-42", "21-43"],          # 18.10.21 - 31.10.21, KW 
    'SS2022': ["22-xxx", "22-xxx"],           # 27.04.22 - 08.05.22, KW 
    'WS2022_23': ["22-42", "22-43"],          # 17.10.22 – 30.10.22, KW
    'SS2023': ["23-16", "23-17"],             # 17.04.23 – 30.04.23, KW
    'WS2023_24': ["23-42", "23-43"],          # 16.10.23 – 29.10.23, KW
    'SS2024': ["24-16", "24-17"],             #
    'WS2024_25': ["24-42", "24-43"]           # 14.10.24 – 27.10.24, KW
}

# Define periods for T2 (Gruppenarbeitsphase 1)
period_2_arr = {
    'SS2021': ["21-18", "21-19", "21-20"],    # 09.05.21 - 29.05.21, KW 18-21
    'WS2021_22': ["21-44", "21-45", "21-46"], # 01.11.21 – 21.11.21, KW 44-46
    'SS2022': ["22-19", "22-20", "22-21"],    # 09.05.22 - 29.05.22, KW 19-21
    'WS2022_23': ["22-44", "22-45", "22-46"], # 31.10.22 – 20.11.22, KW 44-46
    'SS2023': ["22-18", "22-19", "22-20"],    # 01.05.23 – 21.05.23, KW 18-20
    'WS2023_24': ["23-44", "23-45", "23-46"], # 30.10.23 – 19.11.23, KW 44-46
    'SS2024': ["24-18", "24-19", "24-20"],    # 
    'WS2024_25': ["24-44", "24-45", "24-46"]  # 28.10.24 – 17.11.24, KW 44-46
}

# Define periods for T3 (Gruppenarbeitsphase 2)
period_3_arr = {
    'SS2021': ["21-21", "21-22", "21-23", "21-24"], # 30.05.21 - 19.06.21, KW 21-24 
    'WS2021_22': ["21-47", "21-48", "21-49"],       # 22.11.21 – 12.12.21, KW 47-49
    'SS2022': ["22-22", "22-23", "22-24"],          # 30.05.22 - 19.06.22, KW 22-24
    'WS2022_23': ["22-47", "22-48", "22-49"],       # 21.11.22 – 11.12.22, KW 47-49
    'SS2023': ["22-21", "22-22", "22-23"],          # 22.05.23 – 11.06.23, KW 21-23
    'WS2023_24': ["23-47", "23-48", "23-49"],       # 20.11.23 – 10.12.23, KW 47-49
    'SS2024': ["24-21", "24-22", "24-23"],          # 
    'WS2024_25': ["24-47", "24-48", "24-49"]        # 18.11.24 – 08.12.24, KW 47-49
}


# Helper function to retrieve periods
def get_period(semester, period_arr):
    return period_arr.get(semester, [])

# Example usage

period_1 = get_period(semester, period_1_arr)
period_2 = get_period(semester, period_2_arr)
period_3 = get_period(semester, period_3_arr)

# Print the results
#print("Period 1:", period_1)
#print("Period 2:", period_2)
#print("Period 3:", period_3)

print(".. settings loaded")