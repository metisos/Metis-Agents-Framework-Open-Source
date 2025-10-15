#!/usr/bin/env python3
"""
Environment file encryption/decryption utilities for MetisOS
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EnvCrypto:
    """Utility class for encrypting and decrypting .env files."""
    
    @staticmethod
    def generate_key(password, salt=None):
        """Generate a Fernet key from a password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        elif isinstance(salt, str):
            salt = salt.encode()
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    @staticmethod
    def encrypt_file(source_path, target_path, password):
        """Encrypt a .env file with a password."""
        # Generate a random salt
        key, salt = EnvCrypto.generate_key(password)
        fernet = Fernet(key)
        
        # Read the source file
        with open(source_path, 'rb') as file:
            data = file.read()
        
        # Encrypt the data
        encrypted_data = fernet.encrypt(data)
        
        # Write the encrypted data
        with open(target_path, 'wb') as file:
            file.write(encrypted_data)
        
        # Write the salt to a separate file
        salt_path = f"{target_path}.salt"
        with open(salt_path, 'wb') as file:
            file.write(salt)
        
        print(f"Encrypted {source_path} to {target_path}")
        return target_path, salt_path
    
    @staticmethod
    def decrypt_file(source_path, target_path, password):
        """Decrypt an encrypted .env file with a password."""
        # Load the salt
        salt_path = f"{source_path}.salt"
        try:
            with open(salt_path, 'rb') as file:
                salt = file.read()
        except FileNotFoundError:
            raise ValueError(f"Salt file not found: {salt_path}. Cannot decrypt without the salt file.")
        
        # Generate encryption key from password and salt
        key, _ = EnvCrypto.generate_key(password, salt)
        fernet = Fernet(key)
        
        # Read the encrypted file
        with open(source_path, 'rb') as file:
            encrypted_data = file.read()
        
        try:
            # Decrypt the data
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Write the decrypted data
            with open(target_path, 'wb') as file:
                file.write(decrypted_data)
                
            print(f"Decrypted {source_path} to {target_path}")
            return target_path
            
        except Exception as e:
            raise ValueError(f"Decryption failed. Incorrect password or corrupted file: {e}")
