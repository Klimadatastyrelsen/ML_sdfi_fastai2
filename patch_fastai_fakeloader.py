"""Patch fastai _FakeLoader to add in_order for PyTorch 2.6+ DataLoader compatibility."""
import fastai.data.load as m

p = m.__file__
with open(p) as f:
    c = f.read()

old = "store_attr('d,pin_memory,num_workers,timeout,persistent_workers,pin_memory_device')"
new = old + "\n        self.in_order = True"

if old in c and "self.in_order" not in c:
    c = c.replace(old, new, 1)
    with open(p, "w") as f:
        f.write(c)
    print("Patched fastai _FakeLoader: added self.in_order = True")
else:
    print("Patch not needed or already applied")
