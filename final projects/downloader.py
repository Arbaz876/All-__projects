from pytube import YouTube

link = YouTube("https://youtu.be/pHNbm-4reIc?t=1325")

video = link.streams.get_highest_resolution()

video.download()