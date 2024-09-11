import re


def validate_extracted_text(field_label, text,registration_number):

    if field_label == 'Image_ID':
        return validate_Image_ID(text)
    
    if field_label == 'Registration No':
        registration_number = validate_registration_number(text)
        return registration_number
    
    if field_label == 'Chassis No':
        return validate_chassis_number(text)
    
    if field_label == 'Special Notes':
        return validate_special_notes(text)
    
    if field_label == 'Engine No.':
        return validate_engine_number(text)
    
    if field_label == 'Cylinder Capacity':
        return validate_cylinder_capacity(text)
    
    if field_label == 'Class of Vehicle':
        return validate_class_of_vehicle(text)
    
    if field_label == 'Status when Registered':
        return validate_state_when_registered(text)
    
    if field_label == 'Color':
        return validate_color(text)
    
    if field_label == 'Seating Capacity':
        return validate_seating_capacity(text)
    
    if field_label == 'grossWeight':
        return validate_gross_weight(text)
    
    if field_label == 'Lenght , Width , Height':
        return validate_Lenght_Width_Height(text)
    
    if field_label == 'provincialCouncil':
        return validate_provincialCouncil(text,registration_number)
    
    if field_label == 'dateOfFirstRegistration':
        return validate_date_of_first_registration(text)
    
    if field_label == 'Fuel Type':
        return validate_fuel_type(text)
    
    if field_label == 'Type of Body':
        return validate_type_of_body(text)
    
    if field_label == 'Country of Origin':
        return validate_country_of_origin(text)
    
    if field_label == 'Make':
        return validate_make(text)
    
    if field_label == 'Model':
        return validate_model(text)
    
    return text

def validate_model(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]{10,20}$', '', text)
    return cleaned_text.strip()

def validate_make(text):
    
    Make= ["HERO", "Toyota", "Honda", "Nissan", "Suzuki", "Mitsubishi"]

    for make in Make:
        if make in text:
            return make
    return ''

def validate_Image_ID(text):        
    cleaned_text = re.sub(r'[^A-Za-z0-9:.\s]', '', text)
    return cleaned_text.strip()


def validate_registration_number(text):
    
    pattern = r'^[A-Z]{2} [A-Z]{2,3}-\d{4}$'

    if re.match(pattern, text):
        return text
    
    return ''

def validate_chassis_number(text):
    
    pattern = r'^[A-Z0-9\s-]{17,19}$'

    if re.match(pattern, text):
        return text
    
    return ''

def validate_special_notes(text):
    
    pattern = r'^Original$'

    if re.match(pattern, text):
        return text
    
    return 'Original'

def validate_engine_number(text):
    pattern = r'^[A-Z0-9\s-]{10,11}$'

    if re.match(pattern, text):
        return text
    
    return ''

def validate_cylinder_capacity(text):
    
    pattern = r'(\d{1,5})'
   
    match = re.findall(pattern, text)
   
    if match:
        return f"{match[0]}.00 CC"
    
    return ''

def validate_class_of_vehicle(text):

    pattern = r'^[A-Za-z\s]{5,25}$'

    if re.match(pattern, text):
        return text.strip()
    
    return ''

def validate_state_when_registered(text):
    
    pattern = r'^[A-Za-z\s]{5,15}$'
    
    if re.match(pattern, text):
        return text.strip()
               
    return "Brand New"

def validate_color(text):
        
    pattern = r'^[A-Za-z\s]{3,20}$'
        
    if re.match(pattern, text):
        return text.strip()
        
    return ''

def validate_seating_capacity(text):
    
    pattern = r'^\(Ex/Driver\) \d+$'
   
    if re.match(pattern, text):
        return text.strip()  
      
    return '(Ex/Driver)'


def validate_gross_weight(text):
    pattern = r'(\d{1,6}\.\d{2})'
   
    match = re.search(pattern, text)
   
    if match:
        return f"{match.group(1)} KG"
       
    return 'KG'

def validate_Lenght_Width_Height(text):
    pattern = r'(\d{1,3})'
    
    match = re.findall(pattern, text)
    
    # Initialize the default values
    length = "CM"
    width = "CM"
    height = "CM"

    # Assign values based on the number of matches found
    if len(match) > 0:
        length = f"{match[0]} CM"
    if len(match) > 1:
        width = f"{match[1]} CM"
    if len(match) > 2:
        height = f"{match[2]} CM"

    # Return the formatted string with available values
    return f"{length} {width} {height}"


def validate_provincialCouncil(text, registration_number):
    # List of valid provinces
    provinces = ["Western", "Central", "Southern", "Northern", "Eastern", 
                 "North-Western", "North-Central", "Uva", "Sabaragamuwa"]
    
    # Map based on first two letters of the registration number
    province_mapping = {
        "WP": "Western",
        "CP": "Central",
        "SP": "Southern",
        "NP": "Northern",
        "EP": "Eastern",
        "NW": "North-Western",
        "NC": "North-Central",
        "UP": "Uva",
        "SB": "Sabaragamuwa"
    }

    # Check if the extracted text contains any province from the array
    for province in provinces:
        if province in text:
            return province    
    

    # If the registration number is available, extract the first two letters
    if registration_number and len(registration_number) >= 2:
        first_two_letters = registration_number[:2]
        return province_mapping.get(first_two_letters, "Western")
    
    # If no province is found, return "Western Province" by default
    return "Western"


def validate_date_of_first_registration(text):
    
    pattern = r'^\d{2}/\d{2}/\d{4}$'
   
    if re.match(pattern, text):
        return text.strip()
    
    return ''

def validate_fuel_type(text):
        
    pattern = r'^[A-Za-z]{3,8}$'
        
    if re.match(pattern, text):
        return text.strip()
        
    return ''

def validate_type_of_body(text):
    pattern = r'^[A-Za-z\s]{3,15}$'
        
    if re.match(pattern, text):
        return text.strip()
        
    return ''

def validate_country_of_origin(text):
    
    
    Countries= ["India", "Japan", "Germany", "South Korea", "USA", "UK", "Italy", "France", "China"]    
    

    for constry in Countries:
        if constry in text:
            return constry
        
    return ''