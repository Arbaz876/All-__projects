num_1= float(input("enter num 1="))
num_2= float(input("enter num 2="))

choice= input('enter your choice + - / * =')

if choice == '+':
    print(f'Addition :{num_1 + num_2} ' )
    
elif choice == '-':

    print(f'subtraction: {num_1 - num_2}')
    
elif choice =='/':
    print(f'divide : {num_1 / num_2}')
    
elif choice == '*':
    print(f'multiply : {num_1 * num_2}')
    
else :
    print('invalid')
    