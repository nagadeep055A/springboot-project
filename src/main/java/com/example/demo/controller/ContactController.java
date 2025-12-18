package com.example.demo.controller;

import com.example.demo.entity.ContactMessage;
import com.example.demo.service.ContactMessageService;
import java.util.List;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/contact")
@CrossOrigin(origins = "http://localhost:3000")
public class ContactController {

    private final ContactMessageService service;

    public ContactController(ContactMessageService service) {
        this.service = service;
    }

    @PostMapping("/submit")
    public ContactMessage submit(@RequestBody ContactMessage message) {
        return service.saveMessage(message);
    }

    @GetMapping("/messages")
    public List<ContactMessage> getMessages() {
        return service.getAllMessages();
    }
}
