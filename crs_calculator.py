def calculate_exact_crs_score(questionnaire_data):
    """
    Calculate CRS score based on official criteria
    Expected questionnaire_data should be a dictionary with these keys:
    - age: int
    - education: str
    - first_language: dict with 'speaking', 'listening', 'reading', 'writing' CLB levels
    - work_experience: int (years)
    - canadian_work_experience: int (years)
    - education_in_canada: bool
    - arranged_employment: bool
    - provincial_nomination: bool
    - spouse: bool
    - spouse_education: str (if applicable)
    - spouse_language: dict (if applicable)
    - spouse_work_experience: int (if applicable)
    """
    score = 0
    
    # Core / Human Capital Factors (max 460 points)
    
    # Age (max 110 points)
    age_points = {
        18: 90, 19: 95, 20: 100, 21: 100, 22: 100, 23: 100, 24: 100, 25: 100, 26: 100, 27: 100, 
        28: 100, 29: 100, 30: 95, 31: 90, 32: 85, 33: 80, 34: 75, 35: 70, 36: 65, 37: 60, 
        38: 55, 39: 50, 40: 45, 41: 40, 42: 35, 43: 30, 44: 25, 45: 20, 46: 15, 47: 10, 48: 5, 49: 0
    }
    score += age_points.get(questionnaire_data['age'], 0)
    if questionnaire_data.get('arranged_employment'):
        score += 50
    if questionnaire_data.get('education_in_canada'):
        score += 30

    # Canadian Work Experience
    canadian_exp_points = {
        1: 40, 2: 53, 3: 64, 4: 72, 5: 80
    }
    score += canadian_exp_points.get(min(questionnaire_data.get('canadian_work_experience', 0), 5), 0)

    return score

def calculate_crs_score(state):
    questionnaire = state["questionnaire"]
    
    # First, use LLM to extract structured data from questionnaire
    extract_data_prompt = ChatPromptTemplate.from_template("""
    Extract the following information from the questionnaire in a structured format:
    - Age
    - Education level
    - Language test scores (CLB levels)
    - Years of work experience
    - Canadian work experience
    - Education in Canada (yes/no)
    - Arranged employment (yes/no)
    - Provincial nomination (yes/no)
    
    Questionnaire: {questionnaire}
    """)
    
    chain = extract_data_prompt | llm_crs_score | StrOutputParser()
    structured_data = chain.invoke({"questionnaire": questionnaire})
    
    # Parse the structured data and calculate exact score
    # You'll need to add logic to parse the LLM's response into the required format
    parsed_data = parse_llm_response(structured_data)  # You'll need to implement this
    
    exact_score = calculate_exact_crs_score(parsed_data)
    state["crs_score"] = str(exact_score)
    return state

def parse_questionnaire_input():
    """Helper function to gather and structure questionnaire data from user input"""
    questionnaire_data = {}
    
    questionnaire_data['age'] = int(input("Enter your age: "))
    
    education_map = {
        '1': 'secondary',
        '2': 'one_year_degree',
        '3': 'two_year_degree',
        '4': 'three_year_degree',
        '5': 'two_or_more_degrees',
        '6': 'masters',
        '7': 'phd'
    }
    print("\nSelect your highest level of education:")
    print("1. Secondary diploma (high school)")
    print("2. One-year degree/diploma/certificate")
    print("3. Two-year program")
    print("4. Bachelor's degree or three year program")
    print("5. Two or more degrees (one being 3+ years)")
    print("6. Master's degree")
    print("7. Doctoral degree (Ph.D.)")
    edu_choice = input("Enter the number of your choice: ")
    questionnaire_data['education'] = education_map.get(edu_choice, 'secondary')

    # Language scores
    questionnaire_data['first_language'] = {}
    print("\nEnter your CLB levels for first language (4-10):")
    for skill in ['speaking', 'listening', 'reading', 'writing']:
        questionnaire_data['first_language'][skill] = int(input(f"{skill.capitalize()}: "))

    questionnaire_data['work_experience'] = int(input("\nEnter your years of work experience: "))
    questionnaire_data['canadian_work_experience'] = int(input("Enter your years of Canadian work experience: "))
    
    questionnaire_data['education_in_canada'] = input("\nDo you have education from Canada? (yes/no): ").lower() == 'yes'
    questionnaire_data['arranged_employment'] = input("Do you have arranged employment in Canada? (yes/no): ").lower() == 'yes'
    questionnaire_data['provincial_nomination'] = input("Do you have a provincial nomination? (yes/no): ").lower() == 'yes'

    return questionnaire_data


    print("Welcome to the CRS Calculator")
    print("-----------------------------")
    
    questionnaire_data = parse_questionnaire_input()
    total_score = calculate_exact_crs_score(questionnaire_data)
    
    print(f"\nYour estimated CRS score is: {total_score}")

if __name__ == "__main__":
    main()