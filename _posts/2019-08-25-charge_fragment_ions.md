---
layout: post
title: "Charge states of fragment ions in MS2 scan"
categories: proteomics
comments: true
---

## MS2 scan에서 fragment ion들의 charge state는 얼마일까?

MS1 scan에서 precurson ion의 charge state는 decharging step을 통해서 결정된다.  
그렇다면 MS2 scan에서 fragment ion들은 어떤 charge state을 갖게 될까?  

Precursor ion이 +z라는 charge state을 가질 경우에 (positive mode라고 가정하자), fragment ion들은
+1에서 +(z-1) 사이의 charge state을 갖게 된다 (아래 그림 참조).  
![Image of fragmentation](/assets/img/proteomics/fragmentation_20190825.png)

이론적으로 +z charge와 0 charge (i.e. neutral)를 갖는 fragment ion도 생성될 수 있으나 
neutral fragment는 ionization되지 않아서 검출되지 않으므로 고려 대상에서 제외된다.
