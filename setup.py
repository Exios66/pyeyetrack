from setuptools import setup, find_packages

setup(
  name = 'PyEyeTrack',         
  packages = find_packages(),   
  package_data={'': ['shape_predictor_68_face_landmarks.dat']},
  dependency_links = ['http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'],
  include_package_data=True,
  version = '1.0.1',      
  license='MIT',        
  description = 'PyEyeTrack is a python-based pupil-tracking library. The library tracks eyes with the commodity webcam \
                and gives a real-time stream of eye coordinates. It provides the functionality of eye-tracking and \
                blink detection and encapsulates these in a generic interface that allows clients to use these \
                functionalities in a variety of use-cases.', 
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  author = 'Kanchan Sarolkar, Kimaya Badhe, Neha Chaudhari, Samruddhi Kanhed and Shrirang Karandikar',                   
  author_email = 'pyeyetrack@gmail.com',      
  url = 'https://github.com/algoasylum/PyEyeTrack',  
  download_url = 'https://github.com/algoasylum/pyEyeTrack/archive/v_1_0_1.tar.gz',    
  keywords = ['Eye Tracking','blink detection','User Interface','Webcamera'], 
  install_requires=[
  'keyboard>=0.13.5',
  'tqdm>=4.65.0',
  'numpy>=1.19.5',
  'opencv-python>=4.7.0',
  'pandas>=1.2.4',
  'setuptools>=75.0.0',
  'PyQt5>=5.6.0'
  ],  
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)
