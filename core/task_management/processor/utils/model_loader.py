"""
模型加载器
负责加载和管理分析模型
"""
from typing import Dict, Any, Optional
from core.analyzer.detection import YOLODetector
from core.analyzer.segmentation import YOLOSegmentor
from core.tracker import create_tracker
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        """初始化模型加载器"""
        self.loaded_models = {}
        
    async def load_model(self, model_code: str, analysis_type: str) -> Any:
        """
        加载分析模型
        
        Args:
            model_code: 模型代码
            analysis_type: 分析类型（detection/segmentation）
            
        Returns:
            Any: 模型实例
        """
        model_key = f"{model_code}_{analysis_type}"
        
        try:
            if model_key not in self.loaded_models:
                logger.info(f"加载{analysis_type}模型: {model_code}")
                
                if analysis_type == "detection":
                    model = YOLODetector()
                    await model.load_model(model_code)
                elif analysis_type == "segmentation":
                    model = YOLOSegmentor()
                    await model.load_model(model_code)
                else:
                    raise ValueError(f"不支持的分析类型: {analysis_type}")
                    
                self.loaded_models[model_key] = model
                
            return self.loaded_models[model_key]
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
            
    async def create_tracker(self, task_config: Dict[str, Any]) -> Optional[Any]:
        """
        创建目标跟踪器
        
        Args:
            task_config: 任务配置
            
        Returns:
            Optional[Any]: 跟踪器实例
        """
        try:
            if not task_config.get("analysis", {}).get("track_config"):
                return None
                
            track_config = task_config["analysis"]["track_config"]
            tracker_type = track_config.pop("tracker_type", "sort")
            
            logger.info(f"创建跟踪器: {tracker_type}")
            return create_tracker(tracker_type, **track_config)
            
        except Exception as e:
            logger.error(f"创建跟踪器失败: {str(e)}")
            return None
            
    def unload_model(self, model_code: str, analysis_type: str):
        """卸载模型"""
        model_key = f"{model_code}_{analysis_type}"
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"已卸载模型: {model_key}")
            
    def get_loaded_models(self) -> Dict[str, Any]:
        """获取已加载的模型列表"""
        return self.loaded_models 