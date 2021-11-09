import os
import time
import shutil

import tensorflow as tf

try:
    from tensorflow.python.profiler import profiler_client
except:
    pass

try:
    from google.colab import drive
except:
    pass

class MLEnv():
    def __init__(self):
        self.flush_timer = 0
        self.flush_timeout = 180
        self.is_colab = self.check_colab()
        self.check_hardware()

    @staticmethod
    def check_colab():
        try: # Colab instance?
            from google.colab import drive
            is_colab = True
            get_ipython().run_line_magic('load_ext', 'tensorboard')
            try:
                get_ipython().run_line_magic('tensorflow_version', '2.x')
            except:
                pass
        except: # Not? ignore.
            is_colab = False
            pass
        return is_colab

    # Hardware check:
    def check_hardware(self, verbose=True):
        self.is_tpu = False
        self.tpu_is_init = False
        self.is_gpu = False
        self.tpu_address = None

        if self.is_colab:
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                if verbose is True:
                    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
                self.is_tpu = True
                tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
                state=profiler_client.monitor(tpu_profile_service_address, 100, 2)
                if 'TPU v2' in state:
                    print("WARNING: you got old TPU v2 which is limited to 8GB Ram.")

            except ValueError:
                if verbose is True:
                    print("No TPU available")
                self.is_tpu = False

        for hw in ["CPU", "GPU", "TPU"]:
            hw_list=tf.config.experimental.list_physical_devices(hw)
            if len(hw_list)>0:
                if hw=='TPU':
                    self.is_tpu=True
                if hw=='GPU':
                    self.is_gpu=True
                if verbose is True:
                    print(f"{hw}: {hw_list} {tf.config.experimental.get_device_details(hw_list[0])}") 

        if not self.is_tpu:
            if not self.is_gpu:
                if verbose is True:
                    print("WARNING: You have neither TPU nor GPU, this is going to be very slow!")
            else:
                if verbose is True:
                    print("GPU available")
        else:
            tf.compat.v1.disable_eager_execution()
            if verbose is True:
                print("TPU: eager execution disabled!")

    def mount_gdrive(self, mount_point="/content/drive", root_path="/content/drive/My Drive", verbose=True):
        if self.is_colab is True:
            if verbose is True:
                print("You will now be asked to authenticate Google Drive access in order to store training data (cache) and model state.")
                print("Changes will only happen within Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.")
            if not os.path.exists(root_path):
                # drive.flush_and_unmount()
                drive.mount(mount_point) #, force_remount=True)
                return True, root_path
            if not os.path.exists(root_path):
                print(f"Something went wrong with Google Drive access. Cannot save model to {root_path}")
                return False, None
            else:
                return True, root_path
        else:
            if verbose is True:
                print("You are not on a Colab instance, so no Google Drive access is possible.")
            return False, None

    def init_paths(self, project_name, model_name, model_variant=None, log_to_gdrive=False):
        self.save_model = True
        self.model_save_dir=None
        self.cache_stub=None
        self.weights_file = None
        self.project_path = None
        self.log_path = "./logs"
        self.log_to_gdrive = log_to_gdrive
        self.log_mirror_path = None
        if self.is_colab:
            self.save_model, self.root_path = self.mount_gdrive()
        else:
            self.root_path='.'

        if self.save_model:
            if self.is_colab:
                self.project_path=os.path.join(self.root_path,f"Colab Notebooks/{project_name}")
                if log_to_gdrive is True:
                    self.log_mirror_path = os.path.join(self.root_path,f"Colab Notebooks/{project_name}/logs")
                    print(f"Logs will be mirrored to {self.log_mirror_path}, they can be used with a remote Tensorboard instance.")
            else:
                self.project_path=self.root_path
            if model_variant is None:
                self.model_save_dir=os.path.join(self.project_path,f"{model_name}")
                self.weights_file=os.path.join(self.project_path,f"{model_name}_weights.h5")
            else:
                self.model_save_dir=os.path.join(self.project_path,f"{model_name}_{model_variant}")
                self.weights_file=os.path.join(self.project_path,f"{model_name}_{model_variant}_weights.h5")
            self.cache_stub=os.path.join(self.project_path,'data_cache')
            if self.is_tpu is False:
                print(f"Model save-path: {self.model_save_dir}")
            else:
                print(f"Weights save-path: {self.weights_file}")
            print(f'Data cache file-stub {self.cache_stub}')
        return self.model_save_dir, self.weights_file, self.cache_stub, self.log_path

    def gdrive_log_mirror(self):
        # copy directory self.log_path to self.log_mirror_path
        if self.log_to_gdrive is True:
            if self.log_mirror_path is not None:
                if len(self.log_mirror_path)>4 and self.log_mirror_path[-5:]=='/logs':
                    if os.path.exists(self.log_mirror_path) is True:
                        print(f"Removing old log files from {self.log_mirror_path}")
                        shutil.rmtree(self.log_mirror_path)
                    print(f"Staring tree-copy of files from {self.log_mirror_path}. [This can be astonishingly slow!]")
                    shutil.copytree(self.log_path, self.log_mirror_path)
                    print(f"Tensorboard data mirrored to {self.log_mirror_path}")
                else:
                    print(f"Log-mirror path is not valid: {self.log_mirror_path}, it needs to end with '/logs' as sanity-check")

    def epoch_time_func(self, epoch, log):
        if self.log_to_gdrive is True:
            if time.time() - self.flush_timer > self.flush_timeout:
                self.flush_timer=time.time()
                self.gdrive_log_mirror()

