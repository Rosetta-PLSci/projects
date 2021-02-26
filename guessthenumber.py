import random


# USER
def guess(x):
    randomnumber = random.randint(1, x)
    guess = 0
    while guess != randomnumber:
        guess = int(input(f"GUESS A NUMBER BETWEEN 1 AND {x}: "))
        if guess < randomnumber:
            print("TOO LOW")
        elif guess > randomnumber:
            print("TOO HIGH")
    print(f"YOU SUCCESSED AND FOUND THE {randomnumber}")


# MACHINE
def computerguess(x):
    low = 1
    high = x
    feedback = ""
    while feedback != "c" and low != high:
        if low != high:
            guess = random.randint(low, high)
        else:
            guess = low
        feedback = input(f"Is {guess} too high(H), too low(L) or correct(C): ").lower()
        if feedback == "h":
            high = guess - 1
        elif feedback == "l":
            low = guess + 1
        elif feedback == "c":
            print(f"COMPUTER GUESSED YOUR NUMBER, YOUR NUMBER IS {guess}")
            break
        else:
            print("JUST TYPE H,L or C")
            continue


computerguess(1000)
