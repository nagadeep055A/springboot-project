package com.example.demo.service.impl;

import com.example.demo.entity.AiTool;
import com.example.demo.repository.AiToolRepository;
import com.example.demo.service.AiToolService;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

@Service
public class AiToolServiceImpl implements AiToolService {

    private final AiToolRepository aiToolRepository;

    public AiToolServiceImpl(AiToolRepository aiToolRepository) {
        this.aiToolRepository = aiToolRepository;
    }

    @Override
    public List<AiTool> getAllTools() {
        return aiToolRepository.findAll();
    }

    @Override
    public Optional<AiTool> getToolById(Long id) {
        return aiToolRepository.findById(id);
    }

    @Override
    public AiTool saveTool(AiTool tool) {
        return aiToolRepository.save(tool);
    }

    @Override
    public void deleteTool(Long id) {
        aiToolRepository.deleteById(id);
    }
}
