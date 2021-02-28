import base64
import random

while True:
    print("PRESS '/' FOR EXIT")
    desicion = input("(E)NCODING OR (D)ECODING: ").upper()
    passwordnumber = random.randrange(2341,3989,3)

    if desicion == "E":

        yourstring = input("PLEASE WRITE YOUR SENTENCE TO ENCRYPTE: ")
        print()
        stringbytes = yourstring.encode("utf-8")
        stringbase64 = base64.b64encode(stringbytes)
        stringbase64decode = stringbase64.decode("utf-8")

        print(f"YOUR KEY IS {passwordnumber}")
        print(f"YOUR CODE IS: {stringbase64decode}")
        print()

    elif desicion == "D":

        try:
            passwordforinput = int(input("WHAT IS YOUR KEY: "))
        except:
            print("ENTER AN ACCEPTABLE KEY\n")
            continue

        if passwordforinput % 3 == 1:

            try:
                yourstringtodecode = input("PLEASE WRITE YOUR CODE TO DECRYPTE: ")
                print()
                yournewstring = base64.urlsafe_b64decode(yourstringtodecode)
                yourclearstring = yournewstring.decode("utf-8")
                print(yourclearstring)
                print()
            except Exception:
                print("THAT IS NOT A TRUE VALUE, PLEASE TRY AGAIN!\n")
                continue

        else:
            print("THAT IS NOT AN ACCEPTABLE KEY\n")
            continue

    elif desicion == "/":
        print()
        print("PROJECT HAS BEEN ENDED BY USER")
        break

    else:
        print()
        print("TRY AGAIN AND PRESS (E),(D) OR (/)\n")
        continue
