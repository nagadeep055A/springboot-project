package com.example.demo.service;

import com.example.demo.entity.AiTool;
import java.util.List;
import java.util.Optional;

public interface AiToolService {
    List<AiTool> getAllTools();
    Optional<AiTool> getToolById(Long id);
    AiTool saveTool(AiTool tool);
    void deleteTool(Long id);
}
