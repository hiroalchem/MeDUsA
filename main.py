import os
import random
import shutil
from pathlib import Path

import pandas as pd
import wx
import rstr

from tensorflow.python.keras.models import load_model

from utils import save_preprocessed_images, predict, load_folder, bce_dice_loss, dice_coeff, load_images_and_masks, \
    draw_blob4, count_blob4


class Main(wx.Frame):
    def __init__(self, parent, sid, title):
        wx.Frame.__init__(self, parent, sid, title, size=(600, 360))
        panel = wx.Panel(self, id=sid)

        self.button_labels = ["choose dataset folder", "choose mask output folder", "choose result output folder"]

        # dataset folder
        wx.StaticText(panel, wx.ID_ANY, label="data folder", pos=(10, 10))
        self.raw = wx.TextCtrl(panel, wx.ID_ANY, pos=(10, 30), size=(320, 20))
        choose_button_raw = wx.Button(panel, label=self.button_labels[0], pos=(350, 25))

        # mask output folder
        wx.StaticText(panel, wx.ID_ANY, label="mask output folder", pos=(10, 60))
        self.mask = wx.TextCtrl(panel, wx.ID_ANY, pos=(10, 80), size=(320, 20))
        choose_button_mask_output = wx.Button(panel, label=self.button_labels[1], pos=(350, 75))

        # output folder
        wx.StaticText(panel, wx.ID_ANY, label="result output folder", pos=(10, 110))
        self.out = wx.TextCtrl(panel, wx.ID_ANY, pos=(10, 130), size=(320, 20))
        choose_button_output = wx.Button(panel, label=self.button_labels[2], pos=(350, 125))

        self.texts = [self.raw, self.mask, self.out]

        self.Bind(wx.EVT_BUTTON, self.choose_folder)

        wx.StaticText(panel, wx.ID_ANY, label="model hdf5", pos=(10, 160))
        self.modelfile = wx.TextCtrl(panel, wx.ID_ANY, pos=(10, 180), size=(320, 20))
        choose_button_modelfile = wx.Button(panel, label="choose file", pos=(350, 175))
        choose_button_modelfile.Bind(wx.EVT_BUTTON, self.choose_model_file)

        wx.StaticText(panel, wx.ID_ANY, label="percent of images to output detection points", pos=(10, 215))
        self.ratio = wx.SpinCtrl(panel, wx.ID_ANY, min=0, max=100, pos=(10, 240), size=(120, 35))

        button = wx.Button(panel, label='start', pos=(100, 280))
        button.Bind(wx.EVT_BUTTON, self.analyze)

        self.Show(True)
        self.Centre()

    def choose_folder(self, event):
        """ folder selection event"""

        folder = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
                              message="select folder")

        if folder.ShowModal() == wx.ID_OK:
            folder_path = folder.GetPath()
        folder.Destroy()

        self.texts[self.button_labels.index(event.GetEventObject().GetLabel())].SetValue(folder_path)

    def choose_model_file(self, event):
        """ model file (hdf5) selection event """

        file = wx.FileDialog(self, "Open hdf5 file", wildcard="hdf files (*.hdf5)|*.hdf5",
                             style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if file.ShowModal() == wx.ID_OK:
            file_path = file.GetPath()
        file.Destroy()
        self.modelfile.SetValue(file_path)

    def analyze(self, event):

        temp_dir = rstr.xeger(r'^[0-9]{2}[0-9a-zA-Z0-9]{10}')

        data_dir = self.raw.GetValue()
        out_dir = self.mask.GetValue()

        print(data_dir)

        data_dirs = load_folder(data_dir)

        num_dirs = len(data_dirs)
        print(num_dirs)

        if num_dirs == 0:
            wx.Exit()
        else:
            pass

        # progress dialog
        dlg = wx.ProgressDialog('Processing images', f'0/{num_dirs} folders completed', num_dirs,
                                style=wx.PD_APP_MODAL | wx.PD_SMOOTH | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

        # preprocess

        print('start processing')

        for a, d in enumerate(data_dirs):
            save_preprocessed_images(d, temp_dir)

            # update
            alive, skip = dlg.Update(a+1, f'{a+1}/{num_dirs} folders completed')
            if not alive:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                else:
                    pass
                wx.MessageBox(u'aborted.')
                break
        dlg.Destroy()

        if alive:
            pass

        # start prediction

        model = load_model(self.modelfile.GetValue(), custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff})
        data_dirs = load_folder(os.path.join(temp_dir, 'images'))
        num_dirs = len(data_dirs)
        # progress
        dlg = wx.ProgressDialog('Predicting', f'0/{num_dirs} folders completed', num_dirs,
                                style=wx.PD_APP_MODAL | wx.PD_SMOOTH | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

        for a, d in enumerate(data_dirs):
            predict(d, model, out_dir)
            # update
            alive, skip = dlg.Update(a+1, f'{a+1}/{num_dirs} folders completed')
            if not alive:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                else:
                    pass
                wx.MessageBox(u'aborted.')
                break
        dlg.Destroy()
        if alive:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            else:
                pass

        # detection

        raw_dir = data_dir
        mask_dir = out_dir
        out_dir = self.out.GetValue()
        data_dirs = load_folder(raw_dir)
        ratio = self.ratio.GetValue()*0.01
        h = 0.6
        c = -4
        v = 20
        ft = 'png'

        num_dirs = len(data_dirs)

        if num_dirs == 0:
            wx.Exit()
        else:
            pass

        # sampling dir for output detection result images
        idx_out = sorted(random.sample(range(num_dirs), k=int(num_dirs * ratio)))

        # progress dialog
        dlg = wx.ProgressDialog('detecting', f'0/{num_dirs} folders completed', num_dirs,
                                style=wx.PD_APP_MODAL | wx.PD_SMOOTH | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

        # start detection

        df = pd.DataFrame(columns=['sample_name', 'pred_num'])
        for a, d in enumerate(data_dirs):
            imps = sorted(list(d.glob('./**/*tif')))
            try:
                images, masks = load_images_and_masks(imps, Path(mask_dir), ft)
            except Exception as e:
                print(e)
                continue
            if a in idx_out:
                count = draw_blob4(imps, images, masks, out_dir, h=h, gauss=None, b=31, c=c, d=4, v=v)
            else:
                count = count_blob4(images, masks, h=h, gauss=None, b=31, c=c, d=4, v=v)

            tmp_se = pd.Series([d.name, count], index=df.columns)
            df = df.append(tmp_se, ignore_index=True)

            # update
            alive, skip = dlg.Update(a + 1, f'{a + 1}/{num_dirs} folders completed')
            if not alive:
                wx.MessageBox(u'aborted.')
                df.to_csv(os.path.join(out_dir, 'result.csv'))
                break
        dlg.Destroy()
        if alive:
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, 'result.csv'))
            wx.MessageBox(u'done.')


def main():
    app = wx.App(False)
    Main(None, wx.ID_ANY, "MeDUsA")
    app.MainLoop()


if __name__ == "__main__":
    main()
