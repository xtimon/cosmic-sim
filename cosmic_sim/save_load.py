"""
Модуль для сохранения и загрузки состояний симуляций
"""
import json
import numpy as np
from typing import List, Dict, Any
from .body import Body
from .nbody import NBodySimulator


class SimulationIO:
    """Класс для сохранения и загрузки симуляций"""
    
    @staticmethod
    def save_bodies(bodies: List[Body], filename: str):
        """
        Сохранить список тел в JSON файл
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел
        filename : str
            Имя файла
        """
        data = {
            'bodies': []
        }
        
        for body in bodies:
            body_data = {
                'name': body.name,
                'mass': float(body.mass),
                'position': body.position.tolist(),
                'velocity': body.velocity.tolist(),
                'radius': float(body.radius),
                'color': body.color
            }
            data['bodies'].append(body_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_bodies(filename: str) -> List[Body]:
        """
        Загрузить список тел из JSON файла
        
        Parameters:
        -----------
        filename : str
            Имя файла
            
        Returns:
        --------
        List[Body]
            Список тел
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bodies = []
        for body_data in data['bodies']:
            body = Body(
                name=body_data['name'],
                mass=body_data['mass'],
                position=np.array(body_data['position']),
                velocity=np.array(body_data['velocity']),
                radius=body_data['radius'],
                color=body_data['color']
            )
            bodies.append(body)
        
        return bodies
    
    @staticmethod
    def save_simulation_state(simulator: NBodySimulator, filename: str):
        """
        Сохранить полное состояние симуляции
        
        Parameters:
        -----------
        simulator : NBodySimulator
            Симулятор
        filename : str
            Имя файла
        """
        data = {
            'time': float(simulator.time),
            'bodies': [],
            'history': simulator.history
        }
        
        for body in simulator.bodies:
            body_data = {
                'name': body.name,
                'mass': float(body.mass),
                'position': body.position.tolist(),
                'velocity': body.velocity.tolist(),
                'radius': float(body.radius),
                'color': body.color,
                'trajectory': [pos.tolist() for pos in body.trajectory]
            }
            data['bodies'].append(body_data)
        
        # Сохранить в JSON (история может быть большой)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_simulation_state(filename: str) -> NBodySimulator:
        """
        Загрузить состояние симуляции
        
        Parameters:
        -----------
        filename : str
            Имя файла
            
        Returns:
        --------
        NBodySimulator
            Восстановленный симулятор
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bodies = []
        for body_data in data['bodies']:
            body = Body(
                name=body_data['name'],
                mass=body_data['mass'],
                position=np.array(body_data['position']),
                velocity=np.array(body_data['velocity']),
                radius=body_data['radius'],
                color=body_data['color']
            )
            # Восстановить траекторию
            body.trajectory = [np.array(pos) for pos in body_data.get('trajectory', [])]
            bodies.append(body)
        
        simulator = NBodySimulator(bodies)
        simulator.time = data['time']
        simulator.history = data.get('history', [])
        
        return simulator
    
    @staticmethod
    def export_trajectories_csv(bodies: List[Body], filename: str):
        """
        Экспортировать траектории в CSV файл
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел
        filename : str
            Имя файла
        """
        import csv
        
        # Найти максимальную длину траектории
        max_len = max(len(b.trajectory) for b in bodies) if bodies else 0
        
        if max_len == 0:
            raise ValueError("Нет траекторий для экспорта")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Заголовок
            header = ['Time_Index']
            for body in bodies:
                header.extend([f'{body.name}_x', f'{body.name}_y', f'{body.name}_z'])
            writer.writerow(header)
            
            # Данные
            for i in range(max_len):
                row = [i]
                for body in bodies:
                    if i < len(body.trajectory):
                        row.extend(body.trajectory[i].tolist())
                    else:
                        row.extend([None, None, None])
                writer.writerow(row)

