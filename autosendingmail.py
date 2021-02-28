from smtplib import SMTP
import time

subject = input("SUBJECT: ")
print()
message = input("MESSAGE: ")
print()
content = "Subject: {0}\n\n{1}".format(subject, message)

mymail = input("YOUR MAIL ADDRESS: ")
password = input("YOUR MAIL PASSWORD: ")
print()

maillist = []
while True:

    sendtomail = input("TYPE EMAIL ADDRESSES WHAT YOU WANT TO SEND (PRESS '/' TO COMPLETE YOUR LIST): ")
    print()

    if "@" in sendtomail and "." in sendtomail and sendtomail.count("@") == 1:
        maillist.append(sendtomail)
        continue
    elif sendtomail == "/":
        break
    else:
        print("THAT IS NOT AN EMAIL ADDRESS\n")
        continue

mail = SMTP("smtp.gmail.com", 587)
mail.ehlo()
mail.starttls()
mail.login(mymail, password)

while True:
    for mails in maillist:
        time.sleep(10)
        mail.sendmail(mymail, mails, content.encode("utf-8"))
        print("SENDING")
    break
time.sleep(2)
print("PROCESS IS SUCCESSFULLY DONE")
