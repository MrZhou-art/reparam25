import mitsuba as mi
mi.set_variant("scalar_rgb")

import matplotlib.pyplot as plt

scene = mi.load_file("../scenes/cbox.xml")

image = mi.render(scene, spp=256)

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2)); # approximate sRGB tonemapping

mi.util.write_bitmap("my_first_render.png", image)
mi.util.write_bitmap("my_first_render.exr", image)

