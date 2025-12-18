package com.example.demo.controller;

import com.example.demo.entity.AiTool;
import com.example.demo.service.AiToolService;

import jakarta.transaction.Transactional;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping("/api/aitools")
public class AiToolController {

    private final AiToolService aiToolService;

    public AiToolController(AiToolService aiToolService) {
        this.aiToolService = aiToolService;
    }

    @GetMapping
    public List<AiTool> getAllTools() {
        return aiToolService.getAllTools();
    }

    // Keep transaction open while JSON is written
    @Transactional
    @GetMapping("/{id}")
    public AiTool getToolById(@PathVariable Long id) {
        return aiToolService.getToolById(id)
                .orElseThrow(() -> new RuntimeException("AiTool not found"));
    }

    @PostMapping
    public AiTool createTool(@RequestBody AiTool tool) {
        return aiToolService.saveTool(tool);
    }

    @DeleteMapping("/{id}")
    public void deleteTool(@PathVariable Long id) {
        aiToolService.deleteTool(id);
    }
    
    

}
