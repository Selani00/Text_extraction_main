# Labels
labels= [
    'Image_ID',
    'Registration No',
    'Chassis No',
    'Current Owner and Address',
    'Special Notes',
    # 'Absolute Owner',
    'Engine No.',
    'Cylinder Capacity',
    'Class of Vehicle',
    # 'Taxation class',
    'Status when Registered',
    'Fuel Type',
    'Make',
    'Country of Origin',
    'Model',
    # 'Manufacture Description',
    # 'Wheel Base',
    # 'Over Hang',
    'Type of Body',
    # 'Year of Manufacture',
    'Color',
    # 'Previous Owners',
    'Seating Capacity',
    # 'unladenWeight',
    'grossWeight',
    # 'Front Tyre Size',
    # 'Rear Tyre Size',
    # 'Dual ',
    # 'Singal ',
    'Lenght , Width , Height',
    'provincialCouncil',
    'dateOfFirstRegistration',
    # 'taxesPayable'
    ]

word_lists = {
    "Class of Vehicle": ["Motor Car", "Motor Cycle", "Motor Lorry", "Motor Coach", "Dual Purpose Vehicle"],
    # "Taxation class": ["Private Car", "Motor Cycle", "Three-Wheeler", "Dual Purpose Vehicle", "Heavy Goods Vehicle"],
    # "Manufacture Description" : ["Private Car", "Motor Cycle", "Three-Wheeler", "Dual Purpose Vehicle", "Heavy Goods Vehicle"],
    "Status when Registered": ["Brand New", "Suspended", "Cancelled", "Transferred", "Pending", "De-registered", "Expired"],
    "Fuel Type": ["Petrol", "Diesel", "Electric", "Hybrid", "LPG"],
    "Make": ["HERO", "Toyota", "Honda", "Nissan", "Suzuki", "Mitsubishi"],
    "Country of Origin": ["India", "Japan", "Germany", "South Korea", "USA", "UK", "Italy", "France", "China"],
    "Type of Body": ["Panel Van","Open", "closed", "Dual Purpose Van", "SUV", "Coupe"],
    "Year of Manufacture": ["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
    "Color": ["Silver Mica Metallic", "Sport red black", "White Solid", "White", "Yellow", "Black", "White", "Grey", "Brown", "Orange"],
    "provincialCouncil": ["Western", "Central", "Southern", "Northern", "Eastern", "North-Western", "North-Central", "Uva", "Sabaragamuwa"],
    "Special Notes": ["Original"],
    "Seating Capacity" : ["(Ex/Driver) 5" , "(Ex/Driver)", "(Ex/Driver) 4" ,"(Ex/Driver) 2" ,"(Ex/Driver) 1", "(Ex/Driver) 3"]
}


#  box (x, y, width, height)
boxes = [
    (1500, 0, 1030, 100),  #  Image_ID
    (100, 370, 1000,130),   #  Registration No
    (1260, 370, 1000,130),  #  Chassis No
    (100, 570, 1800, 200),  #  Current Owner and Address
    (100, 810, 1800, 550),  #  Special Notes
    # (100, 1450, 1800, 150),  # Absolute Owner
    (100, 1650, 1000, 50),  # Engine No.
    (1260,1650, 1000, 50),  # Cylinder Capacity.
    (100, 1740, 1000, 60),  # Class of Vehicle
    # (1260, 1740, 1000, 60),  # Taxation class
    (100, 1840, 1000, 60),  # Status when Registered
    (1260, 1840, 1000, 60),  # Fuel Type
    (100, 1940, 1000, 60),  # Make
    (1260, 1940, 1000, 60),  # Country of Origin
    (100, 2040, 1000, 60),  # Model
    # (1260, 2035, 1000, 60),  # Manufacture Description
    # (100, 2140, 1000, 60),  # Wheel Base
    # (1260, 2140, 1000, 60),  # Over Hang
    (100, 2250, 1000, 50),  # Type of Body
    # (1260, 2250, 1000, 50),  # Year of Manufacture
    (100, 2340, 1000, 60),  # Color
    # (1260, 2350, 1200, 1450),  # Previous Owners
    (100, 2440, 1000, 55),  # seatingCapacity
    # (600, 2540, 300, 50),  # unladenWeight
    (900, 2540, 300, 50),  # grossWeight
    # (640, 2590, 250, 70),  # Front Tyre Size
    # (640, 2660, 250, 70),  # Rear Tyre Size
    # (1000, 2590, 200, 70),  # Dual 
    # (1000, 2660, 200, 70),  # Singal  
    (100, 2780, 1000, 50),  # Lenght , Width , Height 
    (100, 2880,1000, 50),  # provincialCouncil
    (100, 2980, 1000, 50),  # dateOfFirstRegistration
    # (100, 3080, 1000, 50),  # taxesPayable    
]