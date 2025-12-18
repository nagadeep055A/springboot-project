package com.example.demo.repository;

import com.example.demo.entity.AiTool;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AiToolRepository extends JpaRepository<AiTool, Long> {
    // add custom queries later if needed
}
