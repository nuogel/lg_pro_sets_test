class LG_DECORTATOR:
    def lg_d_fun(self):
        def lg_fun_d1(fun):
            print("lg's decorator")
            fun()

        return lg_fun_d1

    def lg_d_cls(self):
        def lg_fun_d2(cls):
            print("lg's decorator, class name:\n", cls.__name__)

        return lg_fun_d2


lg_class = LG_DECORTATOR()


@lg_class.lg_d_fun()
def lg_outputfun():
    print('out put function')


@lg_class.lg_d_cls()
class LG_CLASS:
    def lg_outputfun2(self):
        print('out put cls')
