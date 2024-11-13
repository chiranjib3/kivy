from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.scrollview import ScrollView
import cv2
import numpy as np
from kivy.uix.filechooser import FileChooserListView


class ImageEnhancerApp(App):
    def build(self):
            self.image_path = None
            self.upscaled_image = None
            self.super_sharp_image = None
            self.sharpened_image = None
            self.low_light_adjusted_image = None

            layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

            # Buttons for different actions
            upload_btn = Button(text="Upload Image", size_hint=(1, 0.1))
            upload_btn.bind(on_release=self.open_file_chooser)

            show_btn = Button(text="Show Enhanced Images", size_hint=(1, 0.1))
            show_btn.bind(on_release=self.show_images)

            # Buttons for downloading images
            download_upscaled = Button(text="Download Upscaled", size_hint=(1, 0.1))
            download_upscaled.bind(on_release=lambda x: self.choose_format(self.upscaled_image, "Upscaled_Image"))

            download_sharpened = Button(text="Download Enhanced", size_hint=(1, 0.1))
            download_sharpened.bind(on_release=lambda x: self.choose_format(self.sharpened_image, "Enhanced_Sharp_Image"))

            download_super = Button(text="Download Super Enhanced", size_hint=(1, 0.1))
            download_super.bind(on_release=lambda x: self.choose_format(self.super_sharp_image, "Super_Enhanced_Image"))

            download_low_light = Button(text="Download Low Light Adjusted", size_hint=(1, 0.1))
            download_low_light.bind(on_release=lambda x: self.choose_format(self.low_light_adjusted_image, "Low_Light_Enhanced_Image"))

            # Add buttons to the layout
            layout.add_widget(upload_btn)
            layout.add_widget(show_btn)
            layout.add_widget(download_upscaled)
            layout.add_widget(download_sharpened)
            layout.add_widget(download_super)
            layout.add_widget(download_low_light)

            return layout

    def open_file_chooser(self, instance):
        filechooser = FileChooserIconView(filters=["*.jpg", "*.png"], show_hidden=False)
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(filechooser)

        select_btn = Button(text="Select", size_hint=(1, 0.2))
        popup_layout.add_widget(select_btn)
        popup = Popup(title="Select an Image", content=popup_layout, size_hint=(0.9, 0.9))

        def select_file(instance):
            self.image_path = filechooser.selection[0] if filechooser.selection else None
            popup.dismiss()
            if self.image_path:
                self.process_image(self.image_path)

        select_btn.bind(on_release=select_file)
        popup.open()

    def process_image(self, image_path):
        image = cv2.imread(image_path)

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = "ESPCN_x4.pb"  # Update this path with your ESPCN model path
        sr.readModel(model_path)
        sr.setModel("espcn", 4)

        self.upscaled_image = sr.upsample(image)

        enhanced_image = cv2.convertScaleAbs(self.upscaled_image, alpha=1.3, beta=30)
        smoothed_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

        gaussian_blur = cv2.GaussianBlur(enhanced_image, (9, 9), 10.0)
        unsharp_image = cv2.addWeighted(enhanced_image, 1.5, gaussian_blur, -0.5, 0)
        edge_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.super_sharp_image = cv2.filter2D(unsharp_image, -1, edge_kernel)
        self.sharpened_image = cv2.filter2D(smoothed_image, -1, edge_kernel)

        self.low_light_adjusted_image = self.process_low_light(enhanced_image)

    def process_low_light(self, enhanced_image):
        lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)

        a = cv2.subtract(a, 15)
        lab_adjusted_image = cv2.merge((l, a, b))
        color_corrected_image = cv2.cvtColor(lab_adjusted_image, cv2.COLOR_LAB2BGR)

        gaussian_blur = cv2.GaussianBlur(color_corrected_image, (5, 5), 10.0)
        unsharp_image = cv2.addWeighted(color_corrected_image, 1.2, gaussian_blur, -0.2, 0)
        edge_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        super_sharp_image = cv2.filter2D(unsharp_image, -1, edge_kernel)

        gamma_corrected = np.power(super_sharp_image / 255.0, 0.8) * 255
        return gamma_corrected.astype(np.uint8)

    def convert_cv2_to_texture(self, cv2_image):
        buf = cv2.flip(cv2_image, 0).tobytes()  # Changed tostring() to tobytes()
        texture = Texture.create(size=(cv2_image.shape[1], cv2_image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def show_images(self, instance):
        if self.upscaled_image is not None and self.super_sharp_image is not None and self.sharpened_image is not None:
            upscaled_texture = self.convert_cv2_to_texture(self.upscaled_image)
            super_sharp_texture = self.convert_cv2_to_texture(self.super_sharp_image)
            enhanced_sharp_texture = self.convert_cv2_to_texture(self.sharpened_image)
            low_light_texture = self.convert_cv2_to_texture(self.low_light_adjusted_image)

            upscaled_img_widget = KivyImage(texture=upscaled_texture, size_hint=(None, None), size=(300, 300))
            super_sharp_img_widget = KivyImage(texture=super_sharp_texture, size_hint=(None, None), size=(300, 300))
            enhanced_sharp_img_widget = KivyImage(texture=enhanced_sharp_texture, size_hint=(None, None), size=(300, 300))
            low_light_img_widget = KivyImage(texture=low_light_texture, size_hint=(None, None), size=(300, 300))

            upscaled_label = Label(text="Upscaled Image", size_hint=(None, None), size=(300, 30), halign='center', valign='middle')
            super_sharp_label = Label(text="Super Enhanced Image", size_hint=(None, None), size=(300, 30), halign='center', valign='middle')
            enhanced_sharp_label = Label(text="Enhanced Sharp Image", size_hint=(None, None), size=(300, 30), halign='center', valign='middle')
            low_light_label = Label(text="Low Light Enhanced Image", size_hint=(None, None), size=(300, 30), halign='center', valign='middle')

            upscaled_layout = BoxLayout(orientation='vertical', size_hint=(None, None), width=300, spacing=10)
            upscaled_layout.add_widget(upscaled_img_widget)
            upscaled_layout.add_widget(upscaled_label)

            super_sharp_layout = BoxLayout(orientation='vertical', size_hint=(None, None), width=300, spacing=10)
            super_sharp_layout.add_widget(super_sharp_img_widget)
            super_sharp_layout.add_widget(super_sharp_label)

            enhanced_sharp_layout = BoxLayout(orientation='vertical', size_hint=(None, None), width=300, spacing=10)
            enhanced_sharp_layout.add_widget(enhanced_sharp_img_widget)
            enhanced_sharp_layout.add_widget(enhanced_sharp_label)

            low_light_layout = BoxLayout(orientation='vertical', size_hint=(None, None), width=300, spacing=10)
            low_light_layout.add_widget(low_light_img_widget)
            low_light_layout.add_widget(low_light_label)

            scroll_layout = BoxLayout(orientation='horizontal', size_hint=(None, None), width=1200, height=450, spacing=10)
            scroll_layout.add_widget(upscaled_layout)
            scroll_layout.add_widget(super_sharp_layout)
            scroll_layout.add_widget(enhanced_sharp_layout)
            scroll_layout.add_widget(low_light_layout)

            scrollview = ScrollView(size_hint=(1, None), height=500)
            scrollview.add_widget(scroll_layout)

            popup_layout = BoxLayout(orientation='vertical')
            popup_layout.add_widget(scrollview)

            # Add close button
            close_button = Button(text="Close", size_hint=(None, None), size=(100, 50))
            popup_layout.add_widget(close_button)

            popup = Popup(title="Enhanced Images", content=popup_layout, size_hint=(0.9, 0.9))

            # Bind the button to close the popup
            close_button.bind(on_release=popup.dismiss)

            popup.open()
        else:
            print("Images not processed yet.")



    def choose_format(self, image, base_filename):
        if image is None:
            return

        # Create a popup to choose the image format
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Add buttons for JPG and PNG
        jpg_btn = Button(text="Download as JPG", size_hint=(1, 0.2))
        png_btn = Button(text="Download as PNG", size_hint=(1, 0.2))

        layout.add_widget(jpg_btn)
        layout.add_widget(png_btn)

        popup = Popup(title="Choose Image Format", content=layout, size_hint=(0.6, 0.4))

        def save_jpg(instance):
            file_path = base_filename + ".jpg"
            self.save_image(image, file_path, format="JPG")
            self.show_success_popup(file_path)
            popup.dismiss()

        def save_png(instance):
            file_path = base_filename + ".png"
            self.save_image(image, file_path, format="PNG")
            self.show_success_popup(file_path)
            popup.dismiss()

        jpg_btn.bind(on_release=save_jpg)
        png_btn.bind(on_release=save_png)

        popup.open()

    def save_image(self, image, filename, format):
        if image is not None:
            if format == "JPG":
                # Save as JPG with 100% quality (highest)
                cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  
            elif format == "PNG":
                # Save as PNG with no compression (highest quality)
                cv2.imwrite(filename, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  

    def show_success_popup(self, file_path):
        # Display success popup with the saved file path
        success_message = f"{file_path} saved successfully!"
        popup = Popup(title="Image Saved", content=Label(text=success_message), size_hint=(0.6, 0.4))
        popup.open()

    
    def on_request_close(self, *args):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text="Are you sure you want to exit?"))

        yes_button = Button(text="Yes", size_hint=(1, 0.2))
        no_button = Button(text="No", size_hint=(1, 0.2))

        content.add_widget(yes_button)
        content.add_widget(no_button)

        close_popup = Popup(title="Exit Confirmation", content=content, size_hint=(0.5, 0.4))

        # If Yes, close the application
        def confirm_exit(instance):
            close_popup.dismiss()
            App.get_running_app().stop()

        # If No, just dismiss the popup
        def cancel_exit(instance):
            close_popup.dismiss()

        yes_button.bind(on_release=confirm_exit)
        no_button.bind(on_release=cancel_exit)

        close_popup.open()



if __name__ == '__main__':
    Window.bind(on_request_close=lambda *args: None)
    ImageEnhancerApp().run()