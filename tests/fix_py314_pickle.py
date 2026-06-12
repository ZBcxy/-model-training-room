"""
Python 3.14 + datasets/dill pickle 兼容性补丁
sitecustomize.py provides the base fix; this module provides additional runtime patches for already-imported libraries.
"""
import pickle


def _safe_batch(self, items, obj=None):
    try:
        self._batch_setitems(items, obj)
    except TypeError:
        self._batch_setitems(items)


def apply_fix():
    try:
        # 1. _Pickler._batch_setitems → accept optional obj
        _orig = pickle._Pickler._batch_setitems
        def _b(self, items, obj=None): return _orig(self, items, obj)
        pickle._Pickler._batch_setitems = _b

        # 2. save_dict → safe wrapper
        _osd = pickle._Pickler.save_dict
        def _sd(self, obj):
            if self.bin: self.write(pickle.EMPTY_DICT)
            else: self.write(pickle.MARK + pickle.DICT)
            self.memoize(obj)
            _safe_batch(self, obj.items(), obj)
        pickle._Pickler.save_dict = _sd

        # 3. save_reduce → safe wrapper
        _osr = pickle._Pickler.save_reduce
        def _sr(self, func, args, state=None, listitems=None, dictitems=None, obj=None):
            if not self.proto >= 2:
                return _osr(self, func, args, state, listitems, dictitems, obj=obj)
            self.save(func); self.save(args); self.write(pickle.POP_MARK)
            if listitems is not None: _safe_batch(self, listitems, obj)
            if dictitems is not None: _safe_batch(self, dictitems, obj)
            if state is not None: self.save(state)
        pickle._Pickler.save_reduce = _sr

        # 4. Patch datasets' _batch_setitems to accept optional obj
        try:
            import datasets.utils._dill as ds
            _odb = ds.Pickler._batch_setitems
            def _dsb(self, items, obj=None):
                import dill._dill
                return dill._dill.Pickler._batch_setitems(self, items, obj)
            ds.Pickler._batch_setitems = _dsb
        except ImportError:
            pass

        print("✅ Python 3.14 pickle 兼容性补丁已就绪")
        return True
    except Exception as e:
        print(f"⚠️ 补丁应用失败: {e}")
        return False


FIXED = apply_fix()
