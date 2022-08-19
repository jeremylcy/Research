# https://pypi.org/project/moviepy/
# pip install ez_setup
# pip install moviepy

import os
from os import path
import math
import shutil
import time
import natsort

import json
from tkinter import W

from moviepy.editor import *
from moviepy.video.fx.all import freeze


def main():

  directory = os.getcwd()

  # initialization
  with open ("settings.json","r") as jsonfile:
    data=json.load(jsonfile)
    print("read successful")
    #print(data)
    jsonfile.close()

  framerate = 25
  sliceSize = int(data['sliceSize'])  #50 #frames
  overlap = int(data['overlap']) #0 #frames
  sliceGap = sliceSize - overlap #the frame index gap between every frames

  files = os.listdir(directory)
  sortfiles = natsort.natsorted(files)
  #file inside folder
  for filename in sortfiles:
    fn = filename[0:-4]
    falltiming, nonfalltiming = [],[]
    if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mkv"):
      print(os.path.join(directory, filename))
      
      text_file = open('timing_batch_5.txt','r')
      counter = 0
      check = 0  
      start_time, end_time = "",""
      tf = text_file.readlines() 
      last = tf[-1] 
      #line in textfile
      for line in tf: 
        line1 = line.strip()
        tmp = 0
        fnn = fn[3:]
        if line1[0] == 's':
          l1 = line1[3:]

        if fn == line1: 
          check = 1

        elif check != 1 and line[0] == 's' and int(fnn) < int(l1):
          break
        elif check == 1 and line[0] == 's' and int(fnn) < int(l1):
          check = 0
          video = clip.duration  
          x = int(video/60)
          y = int(video - x*60)
          z = str(video - x*60 - y)  
          end_nonfallmin = '0' + str(x); print("nonfallmin = " + end_nonfallmin)
          end_nonfallsec = str(y) ; print("nonfallsec = " + end_nonfallsec)
          end_nonfallsubsec = z[2:4]; print("nonfallsubsec = " + end_nonfallsubsec)
          endtime = end_nonfallmin + end_nonfallsec + end_nonfallsubsec
          nonfalltiming.append([temp,endtime])
          start_nonfallmin = temp[0:2] ; print('start_nonfallmin = ' + start_nonfallmin)
          start_nonfallsec = temp[2:4] ; print('start_nonfallsec = ' + start_nonfallsec)
          start_nonfallsubsec = temp[4:6] ; print('start_nonfallsubsec = ' + start_nonfallsubsec)
          start_nonfallframe = (int(start_nonfallmin) * 60 + int(start_nonfallsec) + int(start_nonfallsubsec)/100) * framerate ; print("start_nonfallframe = " + str(start_nonfallframe))
          end_nonfallframe = video * framerate ; print("end_nonfallframe = " + str(end_nonfallframe))

          for x in range (int(start_nonfallframe),int(end_nonfallframe),sliceGap):
            start_nff = str(float(start_nonfallframe))
            deci_snff = start_nff.index('.') 
            deci_startnonfallframe = '0' + start_nff[deci_snff:] 
            end_nff = str(float(end_nonfallframe))
            deci_enff = end_nff.index('.') 
            deci_endnonfallframe = '0' + end_nff[deci_snff:] 
            if x != int(start_nonfallframe):
              begin = last
              if x + sliceGap > end_nonfallframe: 
                end_eachnonfallframe = framerate * clip.duration ; print("end_eachnonfallframe = " + str(end_eachnonfallframe))
                end_eachnonfalltime = clip.duration ; print("end_eachnonfalltime = " + str(end_eachnonfalltime))
                end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                y = str(format(end_eachnonfalltime - end_eachnonfallmin*60 - end_eachnonfallsec, ".2f")) 
                end_eachnonfallsubsec = y[2:4]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)

                last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)

                start_eachnonfallmin = begin[0:2] ; print('start_eachnonfallmin = ' + start_eachnonfallmin)
                start_eachnonfallsec = begin[3:5] ; print('start_eachnonfallsec = ' + start_eachnonfallsec)
                start_eachnonfallsubsec = begin[6:8]; print('start_eachnonfallsubsec = ' + start_eachnonfallsubsec)

                start_clip = int(start_eachnonfallmin) * 60 + int(start_eachnonfallsec) + float(start_eachnonfallsubsec)/100

                clip1 = (VideoFileClip(filename)
                        .subclip(start_clip,end_eachnonfalltime)
                        .resize(resizealgo(clip.w,clip.h))
                        .set_fps(25))

                clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

              else:
                end_eachnonfallframe = x + float(deci_startnonfallframe) + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                end_eachnonfalltime = end_eachnonfallframe/framerate ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime)) 
                end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))  
                end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec)) 
                y = str(format(end_eachnonfalltime - end_eachnonfallmin*60 - end_eachnonfallsec, ".2f")) 
                end_eachnonfallsubsec = y[2:4]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)

                last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec)

                start_eachnonfallmin = begin[0:2] ; print('start_eachnonfallmin1 = ' + start_eachnonfallmin)
                start_eachnonfallsec = begin[3:5] ; print('start_eachnonfallsec1 = ' + start_eachnonfallsec)
                start_eachnonfallsubsec = begin[6:8]; print('start_eachnonfallsubsec = ' + start_eachnonfallsubsec)

                start_clip = int(start_eachnonfallmin) * 60 + int(start_eachnonfallsec) + float(start_eachnonfallsubsec)/100

                clip1 = (VideoFileClip(filename)
                        .subclip(start_clip,start_clip + sliceGap/framerate)
                        .resize(resizealgo(clip.w,clip.h))
                        .set_fps(25))

                clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

            else:
              begin = str(start_nonfallmin) + '.' + str(start_nonfallsec) + '.' + str(start_nonfallsubsec) 
              
              end_eachnonfallframe = start_nonfallframe + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
              end_eachnonfalltime = end_eachnonfallframe/framerate  ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
              end_eachnonfallmin = int(end_eachnonfalltime/60)   ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
              end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
              y = str(format(end_eachnonfalltime - end_eachnonfallmin*60 - end_eachnonfallsec, ".2f"))
              end_eachnonfallsubsec = y[2:4]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
              
              last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)
              
              start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100

              clip1 = (VideoFileClip(filename)
                      .subclip(start_clip,start_clip + sliceGap/framerate)
                      .resize(resizealgo(clip.w,clip.h))
                      .set_fps(25))

              clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

            fallrange = range (int(start_nonfallframe),int(end_nonfallframe),1)
            thefilename = 'negative '

            thefilename = thefilename + str(x) + ' ' + str(begin) + '-' + str(last) + ' ' + 'test ' + '.gif'
            print("filename=" + thefilename)
            clip1.write_gif(thefilename) 
          break 

        #video timings in textfile
        if check == 1 and line[0] == '-':
            counter+=1
            _,y = line.split()
            start_time,end_time = y.split(',')
            falltiming.append([start_time,end_time])
            if counter == 1: 
              nonfalltiming.append(['000000',start_time])
              start_nonfallframe = 0
              start_nonfalltime = '000000'
              start_nonfallmin = '00' 
              start_nonfallsec = '00' 
              start_nonfallsubsec = '00' 
              end_nonfalltime = start_time
              end_nonfallmin = start_time[0:2] ; print('end_nonfallmin = ' + end_nonfallmin)
              end_nonfallsec = start_time[2:4] ; print('end_nonfallsec = ' + end_nonfallsec)
              end_nonfallsubsec = start_time[4:6] ; print('end_nonfallsubsec = ' + end_nonfallsubsec)
              end_nonfallframe = (int(end_nonfallmin) * 60 + int(end_nonfallsec) + int(end_nonfallsubsec)/100) * framerate ; print("end_nonfallframe = " + str(end_nonfallframe))
              clip = VideoFileClip(filename)

              for x in range (int(start_nonfallframe),int(end_nonfallframe),sliceGap):
                start_nff = str(float(start_nonfallframe))
                deci_nff = start_nff.index('.') 
                deci_startnonfallframe = '0' + start_nff[deci_nff:] 
                if x != int(start_nonfallframe):
                  begin = last
                  if x + sliceGap > end_nonfallframe:
                    end_eachnonfallframe = end_nonfallframe ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                    last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)
                    start_eachnonfallmin = begin[0:2] ; print('start_eachnonfallmin = ' + start_eachnonfallmin)
                    start_eachnonfallsec = begin[3:5] ; print('start_eachnonfallsec = ' + start_eachnonfallsec)
                    start_eachnonfallsubsec = begin[6:8]; print('start_eachnonfallsubsec = ' + start_eachnonfallsubsec)
                    
                    start_clip = int(start_eachnonfallmin) * 60 + int(start_eachnonfallsec) + float(start_eachnonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,end_eachnonfalltime)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)
                  else:
                    end_eachnonfallframe = x + float(deci_startnonfallframe) + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime)) 
                    end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))  
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec)) 
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)

                    last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)

                    start_nonfallmin = begin[0:2] ; print('start_nonfallmin = ' + start_nonfallmin)
                    start_nonfallsec = begin[3:5] ; print('start_nonfallsec = ' + start_nonfallsec)
                    start_nonfallsubsec = begin[6:8]; print('start_nonfallsubsec = ' + start_nonfallsubsec)
                  
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,start_clip + sliceGap/framerate)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)
                
                else:
                  if x + sliceGap > end_nonfallframe:
                    begin = str(start_nonfallmin) + '.' + str(start_nonfallsec) + '.' + str(start_nonfallsubsec) 
                  
                    end_eachnonfallframe = end_nonfallframe; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate  ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60)   ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                    last = str(end_eachnonfallmin).zfill(2)  + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2) 
                    print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                          .subclip(start_clip,end_eachnonfalltime)
                          .resize(resizealgo(clip.w,clip.h))
                          .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                  else:
                    begin = str(start_nonfallmin) + '.' + str(start_nonfallsec) + '.' + str(start_nonfallsubsec) 
                  
                    end_eachnonfallframe = start_nonfallframe + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate  ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60)   ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                  
                    last = str(end_eachnonfallmin).zfill(2)  + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2) 
                    print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
                  
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                  
                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,start_clip + sliceGap/framerate)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                #categorize using range
                fallrange = range (int(start_nonfallframe),int(end_nonfallframe),1)
                thefilename = 'negative '
                thefilename = thefilename + str(x) + ' ' + str(begin) + '-' + str(last) + ' ' + 'test ' + '.gif'
                print("filename=" + thefilename)
                clip1.write_gif(thefilename) 

            else: 
              nonfalltiming.append([temp,start_time])
              start_nonfallmin = temp[0:2] ; print("start_nonfallmin = " + start_nonfallmin)
              start_nonfallsec = temp[2:4] ; print("start_nonfallsec = " + start_nonfallsec)
              start_nonfallsubsec = temp[4:6] ; print("start_nonfallsubsec = " + start_nonfallsubsec)
              start_nonfallframe = (int(start_nonfallmin) * 60 + int(start_nonfallsec) + int(start_nonfallsubsec)/100) * framerate ; print("start_nonfallframe = " + str(start_nonfallframe))
              end_nonfallmin = start_time[0:2] ; print("end_nonfallmin = " + end_nonfallmin)
              end_nonfallsec = start_time[2:4] ; print("end_nonfallsec = " + end_nonfallsec)
              end_nonfallsubsec = start_time[4:6] ; print("end_nonfallsubsec = " + end_nonfallsubsec)
              end_nonfallframe = (int(end_nonfallmin) * 60 + int(end_nonfallsec) + int(end_nonfallsubsec)/100) * framerate ; print("end_nonfallframe = " + str(end_nonfallframe))
              start_nff = str(start_nonfallframe)
              deci_nff = start_nff.index('.') 
              deci_startnonfallframe = '0' + start_nff[deci_nff:] 

              for x in range(int(start_nonfallframe),int(end_nonfallframe),sliceGap):
                start_eachnonfallframe = x + float(deci_startnonfallframe); print("start_eachnonfallframe = " + str(start_eachnonfallframe))
                start_eachnonfalltime = start_eachnonfallframe/framerate ; print("start_eachnonfalltime = " + str(start_eachnonfalltime))
                start_eachnonfallmin = int(start_eachnonfalltime/60) ; print("start_eachnonfallmin = " + str(start_eachnonfallmin))
                start_eachnonfallsec = int(start_eachnonfalltime - start_eachnonfallmin*60) ; print("start_eachnonfallsec = " + str(start_eachnonfallsec))
                s_enft = format(start_eachnonfalltime,".2f")
                y = s_enft.index('.')
                start_eachnonfallsubsec = s_enft[y+1:] ; print("start_eachnonfallsubsec = " + start_eachnonfallsubsec)

                if x + sliceGap < end_nonfallframe:
                  end_eachnonfallframe = x + float(deci_startnonfallframe) + sliceGap ; print("end_eachnonfallframe = " + str(end_eachnonfallframe))
                else:
                  end_eachnonfallframe = end_nonfallframe; print("end_eachnonfallframe = " + str(end_eachnonfallframe))
                end_eachnonfalltime = end_eachnonfallframe/framerate ; print("end_eachnonfalltime = " + str(end_eachnonfalltime))
                end_eachnonfallmin = int(end_eachnonfalltime/60) ; print("end_eachnonfallmin = " + str(end_eachnonfallmin))
                end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print("end_eachnonfallsec = " + str(end_eachnonfallsec))
                e_enft = format(end_eachnonfalltime,".2f") 
                z = e_enft.index('.') 
                end_eachnonfallsubsec = e_enft[z+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)

                begin = str(start_eachnonfallmin).zfill(2) + '.' + str(start_eachnonfallsec).zfill(2) + '.' + str(start_eachnonfallsubsec).zfill(2)
                last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)
                print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
                
                start_clip = int(start_eachnonfallmin) * 60 + int(start_eachnonfallsec) + float(start_eachnonfallsubsec)/100
                print('start_clip for eachnonfall = ' + str(start_clip))
                end_clip = int(end_eachnonfallmin) * 60 + int(end_eachnonfallsec) + float(end_eachnonfallsubsec)/100
                print('end_clip for eachnonfall = ' + str(end_clip))

                clip1 = (VideoFileClip(filename)
                        .subclip(start_clip,end_clip)
                        .resize(resizealgo(clip.w,clip.h))
                        .set_fps(25))

                clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                thefilename = 'negative '
                thefilename = thefilename + str(x) + ' ' + str(begin) + '-' + str(last) + ' ' + 'test ' + '.gif'
                print("filename=" + thefilename)
                clip1.write_gif(thefilename) 
          
            temp = end_time

            start_fall_min = start_time[0:2] ; print('start_fall_min = ' + str(start_fall_min))
            start_fall_sec = start_time[2:4] ; print('start_fall_sec = ' + str(start_fall_sec))
            start_fall_subsec = start_time[4:6] ; print('start_fall_subsec = ' + str(start_fall_subsec))

            end_fall_min = end_time[0:2] ; print('end_fall_min = ' + str(end_fall_min))
            end_fall_sec = end_time[2:4] ; print('end_fall_sec = ' + str(end_fall_sec))
            end_fall_subsec = end_time[4:6] ; print('end_fall_subsec = ' + str(end_fall_subsec))
        
            clip = VideoFileClip(filename)

            start_fall_frame = (int(start_fall_min) * 60 + int(start_fall_sec) + float(start_fall_subsec)/100) * framerate
            end_fall_frame = (int(end_fall_min) * 60 + int(end_fall_sec)  + float(end_fall_subsec)/100) * framerate
            print('start_fall_frame=' + str(start_fall_frame) 
                + '; end_fall_frame=' + str(end_fall_frame))

            for x in range (int(start_fall_frame),int(end_fall_frame),sliceGap):
                begin = start_fall_min.zfill(2) + '.' + start_fall_sec.zfill(2) + '.' + start_fall_subsec.zfill(2)
                last = end_fall_min.zfill(2) + '.' + end_fall_sec.zfill(2) + '.' + end_fall_subsec.zfill(2) 
                print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
              
                start_clip = int(start_fall_min) * 60 + int(start_fall_sec) + float(start_fall_subsec)/100
                print('start_clip = ' + str(start_clip))
                end_clip = int(end_fall_min) * 60 + int(end_fall_sec)  + float(end_fall_subsec)/100
                print('end_clip = ' + str(end_clip))


                clip1 = (VideoFileClip(filename)
                        .subclip(start_clip,end_clip)
                        .resize(resizealgo(clip.w,clip.h))
                        .set_fps(25))

                clip1 = freeze(clip1, t= 'end', total_duration = 2, padding_end = 0)
                #categorize using range
                fallrange = range (int(start_fall_frame),int(end_fall_frame),1)
                if (x in fallrange) or ((x+100) in fallrange):
                    thefilename = 'positive '
                else:
                    thefilename = 'negative '; 

                thefilename = thefilename + str(x) + ' ' + str(begin) + '-' + str(last) + ' ' + 'test ' + '.gif'
                print("filename=" + thefilename)
                clip1.write_gif(thefilename) 
            #last timing in textfile
            if tf[-1] == line: 
              video = clip.duration
              x = int(video/60)
              y = int(video - x*60)
              z = str(video - x*60 - y)
              end_nonfallmin = '0' + str(x); print("nonfallmin = " + end_nonfallmin)
              end_nonfallsec = str(y) ; print("nonfallsec = " + end_nonfallsec)
              end_nonfallsubsec = z[2:4]; print("nonfallsubsec = " + end_nonfallsubsec)
              endtime = end_nonfallmin + end_nonfallsec + end_nonfallsubsec
              nonfalltiming.append([temp,endtime])
              
              start_nonfallmin = temp[0:2] ; print('start_nonfallmin = ' + start_nonfallmin)
              start_nonfallsec = temp[2:4] ; print('start_nonfallsec = ' + start_nonfallsec)
              start_nonfallsubsec = temp[4:6] ; print('start_nonfallsubsec = ' + start_nonfallsubsec)
              start_nonfallframe = (int(start_nonfallmin) * 60 + int(start_nonfallsec) + int(start_nonfallsubsec)/100) * framerate ; print("start_nonfallframe = " + str(start_nonfallframe))
              end_nonfallframe = video * framerate ; print("end_nonfallframe = " + str(end_nonfallframe))

              for x in range (int(start_nonfallframe),int(end_nonfallframe),sliceGap):
                start_nff = str(float(start_nonfallframe))
                deci_nff = start_nff.index('.') 
                deci_startnonfallframe = '0' + start_nff[deci_nff:] 

                if x != int(start_nonfallframe):
                  begin = last
                  if x + sliceGap > end_nonfallframe: 
                    end_eachnonfallframe = end_nonfallframe ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                    last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)
                    start_nonfallmin = begin[0:2] ; print('start_nonfallmin = ' + start_nonfallmin)
                    start_nonfallsec = begin[3:5] ; print('start_nonfallsec = ' + start_nonfallsec)
                    start_nonfallsubsec = begin[6:8]; print('start_nonfallsubsec = ' + start_nonfallsubsec)
                    
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,end_eachnonfalltime)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)
                  else:
                    end_eachnonfallframe = x + float(deci_startnonfallframe) + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60) ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))  
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec)) 
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)

                    last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2)

                    start_nonfallmin = begin[0:2] ; print('start_nonfallmin = ' + start_nonfallmin)
                    start_nonfallsec = begin[3:5] ; print('start_nonfallsec = ' + start_nonfallsec)
                    start_nonfallsubsec = begin[6:8]; print('start_nonfallsubsec = ' + start_nonfallsubsec)
                    
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,start_clip + sliceGap/framerate)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                else:
                  if x + sliceGap > end_nonfallframe:
                    begin = str(start_nonfallmin) + '.' + str(start_nonfallsec) + '.' + str(start_nonfallsubsec) 

                    end_eachnonfallframe = end_nonfallframe; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate  ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60)   ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                    last = str(end_eachnonfallmin).zfill(2)  + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2) 
                    print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100
                    
                    clip1 = (VideoFileClip(filename)
                          .subclip(start_clip,end_eachnonfalltime)
                          .resize(resizealgo(clip.w,clip.h))
                          .set_fps(25))

                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                  else:
                    begin = str(start_nonfallmin) + '.' + str(start_nonfallsec) + '.' + str(start_nonfallsubsec) 
                    
                    end_eachnonfallframe = start_nonfallframe + sliceGap ; print('end_eachnonfallframe = ' + str(end_eachnonfallframe))
                    end_eachnonfalltime = end_eachnonfallframe/framerate  ; print('end_eachnonfalltime = ' + str(end_eachnonfalltime))
                    end_eachnonfallmin = int(end_eachnonfalltime/60)   ; print('end_eachnonfallmin = ' + str(end_eachnonfallmin))
                    end_eachnonfallsec = int(end_eachnonfalltime - end_eachnonfallmin*60) ; print('end_eachnonfallsec = ' + str(end_eachnonfallsec))
                    e_enft = format(end_eachnonfalltime,".2f") 
                    y = e_enft.index('.')
                    end_eachnonfallsubsec = e_enft[y+1:]  ; print('end_eachnonfallsubsec = ' + end_eachnonfallsubsec)
                    
                    last = str(end_eachnonfallmin).zfill(2) + '.' + str(end_eachnonfallsec).zfill(2) + '.' + str(end_eachnonfallsubsec).zfill(2) 
                    print ('x=' + str(x) + '-> ' + begin + ' to ' + last)
                    
                    start_clip = int(start_nonfallmin) * 60 + int(start_nonfallsec) + float(start_nonfallsubsec)/100

                    clip1 = (VideoFileClip(filename)
                            .subclip(start_clip,start_clip + sliceGap/framerate)
                            .resize(resizealgo(clip.w,clip.h))
                            .set_fps(25))
                    clip1 = freeze(clip1, t='end', total_duration = 2, padding_end=0)

                #categorize using range
                fallrange = range (int(start_nonfallframe),int(end_nonfallframe),1)
                thefilename = 'negative '
                thefilename = thefilename + str(x) + ' ' + str(begin) + '-' + str(last) + ' ' + 'test ' + '.gif'
                print("filename=" + thefilename)
                clip1.write_gif(thefilename) 
        else:
            continue
      text_file.close()
    else:
      continue

  time.sleep(2) #wait 5 sec for the last gif to be written

  for filename in os.listdir(directory):
    if filename.endswith(".gif") and filename.startswith('positive '):
      if not os.path.isdir('positive '):
        os.mkdir('positive ')
      shutil.move (filename, directory + '\\positive')
    elif filename.endswith(".gif") and filename.startswith('negative '):
      if not os.path.isdir('negative '):
        os.mkdir('negative ')
      shutil.move (filename, directory + '\\negative') 

def resizealgo(w,h):
  if w < h:
    temp = w
  else:
    temp = h

  r = 224/temp

  return r  

if __name__== "__main__":
   main()

#Reference:
#https://anaconda.org/conda-forge/moviepy
#http://zulko.github.io/blog/2014/01/23/making-animated-gifs-from-video-files-with-python/
#https://www.codespeedy.com/conversion-of-video-to-gif-using-python/
#https://zulko.github.io/moviepy/getting_started/efficient_moviepy.html
#https://www.geeksforgeeks.org/moviepy-fps-of-video-file-clip/
#https://www.geeksforgeeks.org/moviepy-getting-duration-of-video-file-clip/
#https://www.reddit.com/r/moviepy/comments/2bsnrq/is_it_possible_to_get_the_length_of_a_video/
#https://www.reddit.com/r/moviepy/comments/gasill/problems_with_clips_resolution/fp1yv5r/
