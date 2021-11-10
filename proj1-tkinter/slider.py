import tkinter as tk

class PopupSliderButton(tk.Button):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(command=self.toggle)
        self.sliderframe = tk.Frame(self.winfo_toplevel(), bd=1, relief="sunken", bg="#ebebeb")
        self.slider = tk.Scale(self.sliderframe, from_=0, to_=100, background=self.sliderframe.cget("background"))
        self.slider.pack(fill="both", expand=True, padx=4, pady=4)

    def get(self):
        return self.slider.get()

    def toggle(self):
        if self.sliderframe.winfo_viewable():
            self.sliderframe.place_forget()
        else:
            self.sliderframe.place(in_=self, x=0, y=-4, anchor="sw")

root = tk.Tk()
root.geometry("200x300")

text = tk.Text(root, bd=1, relief="sunken", highlightthickness=0)
button_frame = tk.Frame(root)

button_frame.pack(side="bottom", fill="x")
text.pack(side="top", fill="both", expand=True, padx=2, pady=2)

select_button = tk.Button(button_frame, text="select")
pop_up_button = PopupSliderButton(button_frame, text="pop_up")

select_button.pack(side="left")
pop_up_button.pack(side="right")


text.insert("end", "line 3\nline 2\nline1")

root.mainloop()