while True:
    x = input("You: ").strip()
    print("ECHO:", x)
    if x.lower() == "exit":
        print("Bye")
        break
