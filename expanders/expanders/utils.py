def graphviz_image(G, fname=None, inline=True):
    """Routine to print or display graph using graphviz
    (works better for small graphs).
    """
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(G)
    A.layout('dot')
    png = A.draw(fname, format='png')
    from PIL import Image
    import io
    import matplotlib.pyplot as plt
    image = Image.open(io.BytesIO(png))

    if inline:
        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
        ax.imshow(image)
        ax.axis('off')
        plt.show()
    else:
        inage.show()

def profile_cache():
    """Elementary speedup check due to caching in GraphBuilder.
    """
    import sympy
    import time
    from .lps import LPS3

    # Pick a prime number
    p = sympy.prime(100)

    builder = LPS3(p, remove_parallel_edges=False, remove_self_edges=False)
    builder.build()

    start_time = time.time()
    builder.spectrum
    elapsed_time = time.time() - start_time
    print('First call: {:.2f}ms'.format(elapsed_time*1e3))

    start_time = time.time()
    builder.spectrum
    elapsed_time = time.time() - start_time
    print('Second call: {:.2f}ms'.format(elapsed_time*1e3))

    start_time = time.time()
    builder.p = sympy.nextprime(builder.p)
    builder.spectrum
    elapsed_time = time.time() - start_time
    print('First call after changing parameter: {:.2f}ms'.format(elapsed_time*1e3))
