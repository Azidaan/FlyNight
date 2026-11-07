"""Причинные входы программы1/3/4; это не обученная модель и не полный трекер.

Изображения для движения должны заранее находиться на одной сетке координат.
Сопоставление объектов и компенсацию камеры этот модуль не выдумывает.
"""
from dataclasses import dataclass
import numpy as np


def motion_history(images, times, valid, *, query_time, max_age, threshold):
    """Давность реальных изменений на явно совмещённых изображениях.

    images: T,H,W, float [0,1]; valid: T,H,W, bool. Параметры окна и порога
    задаёт будущий протокол, а не скрытые константы этой подготовки.
    """
    pixels=np.asarray(images,dtype=np.float32); t=np.asarray(times,dtype=np.float64)
    mask=np.asarray(valid,dtype=bool)
    if pixels.ndim!=3 or mask.shape!=pixels.shape or t.shape!=(len(pixels),):
        raise ValueError('Нужны изображения и маски T,H,W и времена T')
    if not np.isfinite(pixels).all() or np.any((pixels<0)|(pixels>1)) or not np.isfinite(t).all():
        raise ValueError('Пиксели [0,1] и время должны быть конечными')
    if len(t)>1 and np.any(np.diff(t)<=0):raise ValueError('Времена должны строго возрастать')
    if not np.isfinite(query_time) or not np.isfinite(max_age) or max_age<=0 or not 0<=threshold<=1:
        raise ValueError('Неверное время, окно или порог')
    selected=np.flatnonzero((t<=query_time)&(t>=query_time-max_age))
    history=np.zeros(pixels.shape[1:],np.float32); support=np.zeros_like(history,dtype=bool)
    for previous,current in zip(selected[:-1],selected[1:]):
        usable=mask[previous]&mask[current]
        change=(np.abs(pixels[current]-pixels[previous])>threshold)&usable
        age=np.float32(max(0.,1.-(query_time-t[current])/max_age))
        history[change]=age;support|=usable
    return dict(recency=history,comparison_valid=support,used_times=t[selected].copy())


@dataclass(frozen=True)
class Observation:
    time: float
    box_xywh: np.ndarray
    appearance: np.ndarray
    quality: float


class ObservationHistory:
    """История назначенных РЕАЛЬНЫХ наблюдений одного рабочего ID.

    Рабочий ID не гарантирует истинную личность. Неизвестные метки и обучение
    находятся вне этого контейнера; прогнозы Калмана сюда не добавляются.
    """
    def __init__(self, capacity, feature_dim):
        if capacity<1 or feature_dim<1:raise ValueError('Неверная ёмкость')
        self.capacity=int(capacity);self.feature_dim=int(feature_dim);self._rows=[]

    def append(self, *, time, box_xywh, appearance, quality, observed):
        if observed is not True:raise ValueError('Прогноз не является реальным наблюдением')
        box=np.asarray(box_xywh,dtype=np.float32).copy()
        vector=np.asarray(appearance,dtype=np.float32).copy()
        if box.shape!=(4,) or vector.shape!=(self.feature_dim,):raise ValueError('Неверные формы входов')
        if not np.isfinite(box).all() or np.any(box[2:]<=0) or not np.isfinite(vector).all():
            raise ValueError('Неверная рамка или признаки')
        if not np.isfinite(time) or not np.isfinite(quality) or not 0<=quality<=1:
            raise ValueError('Неверные время/надёжность')
        if self._rows and time<=self._rows[-1].time:raise ValueError('Наблюдения должны поступать по времени')
        norm=np.linalg.norm(vector)
        if norm<=0:raise ValueError('Нулевой признак нельзя нормировать')
        vector/=norm;box.setflags(write=False);vector.setflags(write=False)
        self._rows.append(Observation(float(time),box,vector,float(quality)))
        self._rows=self._rows[-self.capacity:]

    def view(self, *, query_time, max_age, reference_xywh):
        """Только строго прошлые реальные наблюдения; кандидат передаётся отдельно."""
        ref=np.asarray(reference_xywh,dtype=np.float32)
        self._check_query(query_time,max_age)
        if ref.shape!=(4,) or not np.isfinite(ref).all() or np.any(ref[2:]<=0) or max_age<=0:
            raise ValueError('Неверная опорная рамка или окно')
        rows=[r for r in self._rows if query_time-max_age<=r.time<query_time]
        appearances=np.zeros((self.capacity,self.feature_dim),np.float32)
        geometry=np.zeros((self.capacity,4),np.float32)
        ages=np.zeros(self.capacity,np.float32);quality=np.zeros_like(ages);valid=np.zeros(self.capacity,bool)
        center=ref[:2]+ref[2:]/2
        for i,r in enumerate(rows):
            appearances[i]=r.appearance
            geometry[i,:2]=(r.box_xywh[:2]+r.box_xywh[2:]/2-center)/ref[2:]
            geometry[i,2:]=np.log(r.box_xywh[2:]/ref[2:])
            ages[i]=query_time-r.time;quality[i]=r.quality;valid[i]=True
        return dict(appearance=appearances,geometry=geometry,age_seconds=ages,quality=quality,valid=valid,
            observed_times=np.array([r.time for r in rows],np.float64))

    def simple_controls(self, *, query_time, max_age, ema_alpha):
        """Согласованные последний признак и простое экспоненциальное усреднение."""
        self._check_query(query_time,max_age)
        if not 0<ema_alpha<=1:raise ValueError('Неверная сила усреднения')
        rows=[r for r in self._rows if query_time-max_age<=r.time<query_time]
        if not rows:return None,None
        average=rows[0].appearance.copy()
        for r in rows[1:]:average=(1-ema_alpha)*average+ema_alpha*r.appearance
        norm=np.linalg.norm(average)
        if norm>0:average/=norm
        return rows[-1].appearance.copy(),average

    def _check_query(self,query_time,max_age):
        if not np.isfinite(query_time) or not np.isfinite(max_age) or max_age<=0:
            raise ValueError('Неверное время или окно')
        if self._rows and self._rows[-1].time>=query_time:
            raise ValueError('В банке уже есть текущее или будущее наблюдение: порядок исполнения нарушен')
