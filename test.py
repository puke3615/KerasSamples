class A:
    def __call__(self, *args, **kwargs):
        a = args[0]
        print 123, a


def say(*args):
    sum = 0
    for a in args:
        sum += a
    return sum


a = [1, 2, 3]
print say(1, 2, 3)
print say(1)
print say()
print say(*a)
