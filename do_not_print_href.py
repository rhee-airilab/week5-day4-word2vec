# do_not_print_href.py
def load_ipython_extension(ipython):
    from IPython.core.display import display,HTML
    display(HTML('''<style>
@media print {
  a[href]:after {
    content: none !important;
  }
}
</style>'''))

def unload_ipython_extension(ipython):
    pass
